"""
Forward Contribution Propagation (FCP) for Genetic Programming Trees - v2

A post-hoc explanation method that tracks terminal contributions through
GP tree evaluation, providing instance-specific attribution for predictions.

Version 2 Enhancements:
- Signed numeric contributions (SHAP-like, sum to output value)
- Decision-centric tracking (decisions as first-class objects)
- Satisfaction metrics and tipping points for decisions
- Decision space enumeration (all possible paths through the tree)
- Separate tracking of value contributions vs decision contributions

Usage:
    from fcp_tracker_v2 import (
        TrackedValue,
        create_tracked_pset,
        evaluate_with_tracking,
        find_counterfactual,
        explain_prediction,
    )

    # Create primitive set
    pset = create_tracked_pset("my_problem", ["x", "y", "z"])
    pset.addTerminal(TrackedValue.from_constant(1.0), name="ONE")

    # Evaluate with tracking
    result = evaluate_with_tracking(tree, pset, {"x": 1.0, "y": 2.0, "z": 3.0})
    print(result.explain())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from functools import wraps
from enum import Enum
import math
import copy
import operator
import random

from deap import gp, base, creator, tools


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core data structures
    "TrackedValue",
    "SignedContributions",
    "Decision",
    "DecisionPath",
    "DecisionSpace",
    "EvaluationResult",
    # Comparison types
    "ComparisonOp",
    # Primitive operations (for advanced use)
    "TrackedPrimitives",
    # Main functions
    "create_tracked_pset",
    "evaluate_with_tracking",
    "find_counterfactual",
    "compute_flip_sensitivity",
    "explain_prediction",
    "synthesize_explanation",
    # Explanation classes
    "Explanation",
    "ExplanationMode",
    "Factor",
    "Counterfactual",
    # Merging utilities (for custom primitives)
    "merge_signed_additive",
    "merge_signed_multiplicative",
]


# =============================================================================
# Comparison Operator Enum
# =============================================================================


class ComparisonOp(Enum):
    """Comparison operators with their semantic meanings."""

    LT = "<"  # observation < threshold
    LTE = "<="  # observation <= threshold
    GT = ">"  # observation > threshold
    GTE = ">="  # observation >= threshold
    EQ = "=="  # observation == threshold

    @property
    def satisfied_when(self) -> str:
        """Human-readable description of when condition is satisfied."""
        return {
            ComparisonOp.LT: "observation < threshold",
            ComparisonOp.LTE: "observation ≤ threshold",
            ComparisonOp.GT: "observation > threshold",
            ComparisonOp.GTE: "observation ≥ threshold",
            ComparisonOp.EQ: "observation = threshold",
        }[self]


# =============================================================================
# Signed Contributions (SHAP-like, sum to output)
# =============================================================================


@dataclass
class SignedContributions:
    """
    Tracks how each terminal contributes to a value, with sign.

    Properties:
    - contributions[terminal] can be positive or negative
    - sum(contributions.values()) + baseline ≈ value
    - Positive contribution = increases the output
    - Negative contribution = decreases the output
    """

    contributions: Dict[str, float] = field(default_factory=dict)
    baseline: float = 0.0  # The "expected" or "neutral" component

    @property
    def total(self) -> float:
        """Sum of all contributions + baseline."""
        return self.baseline + sum(self.contributions.values())

    @property
    def total_positive(self) -> float:
        """Sum of positive contributions only."""
        return sum(v for v in self.contributions.values() if v > 0)

    @property
    def total_negative(self) -> float:
        """Sum of negative contributions only."""
        return sum(v for v in self.contributions.values() if v < 0)

    def normalized(self) -> Dict[str, float]:
        """Get contributions normalized to sum to 1 (by absolute value)."""
        total_abs = sum(abs(v) for v in self.contributions.values())
        if total_abs == 0:
            return {}
        return {k: v / total_abs for k, v in self.contributions.items()}

    def top_contributors(
        self, n: int = 5, by_absolute: bool = True
    ) -> List[Tuple[str, float]]:
        """Get top N contributors."""
        if by_absolute:
            key = lambda x: abs(x[1])
        else:
            key = lambda x: x[1]
        return sorted(self.contributions.items(), key=key, reverse=True)[:n]

    def copy(self) -> "SignedContributions":
        return SignedContributions(
            contributions=self.contributions.copy(),
            baseline=self.baseline,
        )

    def __repr__(self):
        top = sorted(self.contributions.items(), key=lambda x: -abs(x[1]))[:3]
        contribs = ", ".join(f"{k}: {v:+.3f}" for k, v in top)
        return f"SignedContributions(total={self.total:.4f}, [{contribs}])"


# =============================================================================
# Decision Data Structures (Decision-Centric Tracking)
# =============================================================================


@dataclass
class Decision:
    """
    A single decision point - the atomic unit of decision analysis.

    The decision is treated as a first-class object, not just an aggregation
    of terminal contributions. Observation and threshold are tracked separately.
    """

    # The comparison
    op: ComparisonOp
    observation_value: float  # The value being tested (left side)
    threshold_value: float  # The boundary value (right side)
    observation_source: str  # Expression for observation
    threshold_source: str  # Expression for threshold
    observation_contributors: Dict[str, float] = field(default_factory=dict)
    threshold_contributors: Dict[str, float] = field(default_factory=dict)

    # Branch expressions (what happens on TRUE vs FALSE)
    then_expr: str = ""  # Expression taken when TRUE
    else_expr: str = ""  # Expression taken when FALSE

    # Context
    depth: int = 0  # 0 = outermost (first) decision
    branch_taken: bool = True  # True = condition satisfied
    active: bool = True  # True = this decision was on the active execution path

    # Epsilon for boundary computations
    epsilon: float = 1e-9

    @property
    def result(self) -> bool:
        """The boolean outcome of this decision."""
        if self.op == ComparisonOp.LTE:
            return self.observation_value <= self.threshold_value
        elif self.op == ComparisonOp.LT:
            return self.observation_value < self.threshold_value
        elif self.op == ComparisonOp.GTE:
            return self.observation_value >= self.threshold_value
        elif self.op == ComparisonOp.GT:
            return self.observation_value > self.threshold_value
        elif self.op == ComparisonOp.EQ:
            return abs(self.observation_value - self.threshold_value) < self.epsilon
        return False

    @property
    def raw_margin(self) -> float:
        """Raw distance from boundary. Positive = condition satisfied."""
        if self.op in (ComparisonOp.LTE, ComparisonOp.LT):
            return self.threshold_value - self.observation_value
        elif self.op in (ComparisonOp.GTE, ComparisonOp.GT):
            return self.observation_value - self.threshold_value
        elif self.op == ComparisonOp.EQ:
            return self.epsilon - abs(self.observation_value - self.threshold_value)
        return 0.0

    @property
    def satisfaction(self) -> float:
        """
        Normalized satisfaction score.

        Positive = condition satisfied with margin
        Zero = exactly on boundary
        Negative = condition not satisfied
        """
        diff = self.observation_value - self.threshold_value
        scale = max(abs(self.threshold_value), abs(self.observation_value), 1.0)

        if self.op in (ComparisonOp.LT, ComparisonOp.LTE):
            return -diff / scale
        elif self.op in (ComparisonOp.GT, ComparisonOp.GTE):
            return diff / scale
        else:  # EQ
            return -abs(diff) / scale if scale > 0 else 0.0

    @property
    def tipping_point(self) -> float:
        """The observation value that would flip the decision."""
        t = self.threshold_value
        eps = self.epsilon

        if self.op == ComparisonOp.LT:
            return t
        elif self.op == ComparisonOp.LTE:
            return t + eps
        elif self.op == ComparisonOp.GT:
            return t
        elif self.op == ComparisonOp.GTE:
            return t - eps
        else:  # EQ
            return t + eps if self.result else t

    @property
    def distance_to_flip(self) -> float:
        """How much the observation would need to change to flip the decision."""
        return self.tipping_point - self.observation_value

    @property
    def satisfaction_description(self) -> str:
        """Human-readable satisfaction level."""
        s = self.satisfaction
        if s > 0.5:
            return "comfortably satisfied"
        elif s > 0.1:
            return "satisfied with margin"
        elif s > 0:
            return "barely satisfied"
        elif s == 0:
            return "exactly on boundary"
        elif s > -0.1:
            return "barely not satisfied"
        elif s > -0.5:
            return "not satisfied"
        else:
            return "far from satisfied"

    @property
    def expression(self) -> str:
        """String representation of the comparison."""
        return f"{self.observation_source}:{self.observation_value:.4g} {self.op.value} {self.threshold_source}:{self.threshold_value:.4g}"

    @property
    def power(self) -> float:
        """Decision power at this depth (0.5^(depth+1))."""
        return 0.5 ** (self.depth + 1)

    def describe(self) -> str:
        """Human-readable description of this decision."""
        branch = "TRUE (satisfied)" if self.branch_taken else "FALSE (not satisfied)"
        sat_desc = self.satisfaction_description

        lines = [
            f"Decision[{self.depth}]: {self.observation_source} {self.op.value} {self.threshold_source}",
            f"  Values: {self.observation_value:.4g} {self.op.value} {self.threshold_value:.4g}",
            f"  Result: {branch}",
            f"  Satisfaction: {self.satisfaction:.3f} ({sat_desc})",
            f"  Tipping point: {self.tipping_point:.4g}",
            f"  Distance to flip: {self.distance_to_flip:.4g}",
            f"  Decision power: {self.power:.1%}",
        ]

        # Show contributors
        obs_contribs = ", ".join(
            f"{k}: {v:.0%}"
            for k, v in self.observation_contributors.items()
            if not k.startswith("_")
        )
        thr_contribs = ", ".join(
            f"{k}: {v:.0%}"
            for k, v in self.threshold_contributors.items()
            if not k.startswith("_")
        )
        if obs_contribs:
            lines.append(f"  Observation from: {obs_contribs}")
        if thr_contribs:
            lines.append(f"  Threshold from: {thr_contribs}")

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"Decision[{self.depth}]({self.observation_value:.4g} "
            f"{self.op.value} {self.threshold_value:.4g} "
            f"→ {'T' if self.branch_taken else 'F'}, power={self.power:.1%})"
        )


@dataclass
class DecisionPath:
    """
    A complete path through the decision tree.

    Represents one possible "reality" - a specific sequence of decisions
    leading to a specific output.
    """

    decisions: List[Decision] = field(default_factory=list)
    output_value: float = 0.0
    output_contributors: Dict[str, float] = field(default_factory=dict)
    output_signed: Optional[SignedContributions] = None
    output_expr: str = ""
    is_actual: bool = False

    @property
    def total_power(self) -> float:
        """Total decision power for this path: 0.5^n for n decisions."""
        if not self.decisions:
            return 1.0
        return 0.5 ** len(self.decisions)

    @property
    def signature(self) -> str:
        """Compact string representation of the path."""
        if not self.decisions:
            return f"→ {self.output_expr or self.output_value}"

        parts = []
        for d in self.decisions:
            branch = "T" if d.branch_taken else "F"
            parts.append(
                f"({d.observation_source}{d.op.value}{d.threshold_source}):{branch}"
            )

        return " → ".join(parts) + f" → {self.output_expr or self.output_value}"

    @property
    def branch_sequence(self) -> Tuple[bool, ...]:
        """Tuple of branch outcomes (for comparison between paths)."""
        return tuple(d.branch_taken for d in self.decisions)

    def get_critical_decision(self) -> Optional[Decision]:
        """Get the decision closest to flipping (smallest relative distance)."""
        if not self.decisions:
            return None
        scale_fn = lambda d: max(abs(d.threshold_value), 1.0)
        return min(self.decisions, key=lambda d: abs(d.distance_to_flip) / scale_fn(d))

    def __repr__(self):
        marker = " [ACTUAL]" if self.is_actual else ""
        return f"Path({self.signature}, power={self.total_power:.1%}){marker}"


@dataclass
class DecisionSpace:
    """
    The complete decision space - all possible paths through the tree.
    """

    paths: List[DecisionPath] = field(default_factory=list)

    @property
    def actual_path(self) -> Optional[DecisionPath]:
        """The path that was actually taken."""
        for p in self.paths:
            if p.is_actual:
                return p
        return None

    @property
    def total_power(self) -> float:
        """Should sum to 1.0 if all paths enumerated."""
        return sum(p.total_power for p in self.paths)

    @property
    def output_distribution(self) -> Dict[float, float]:
        """Map of output value → probability (decision power)."""
        dist = {}
        for p in self.paths:
            dist[p.output_value] = dist.get(p.output_value, 0) + p.total_power
        return dist

    def get_counterfactuals(self) -> List[DecisionPath]:
        """All paths that weren't taken."""
        return [p for p in self.paths if not p.is_actual]

    def get_single_flip_counterfactuals(self) -> List[Tuple[int, DecisionPath]]:
        """
        Paths that differ by exactly one decision.

        Returns: List of (decision_index, path) tuples.
        """
        actual = self.actual_path
        if not actual:
            return []

        actual_seq = actual.branch_sequence
        results = []

        for path in self.paths:
            if path.is_actual:
                continue

            path_seq = path.branch_sequence

            # Find differences
            if len(path_seq) != len(actual_seq):
                min_len = min(len(path_seq), len(actual_seq))
                diffs = [i for i in range(min_len) if path_seq[i] != actual_seq[i]]
                if len(diffs) == 1 and diffs[0] == min_len - 1:
                    results.append((diffs[0], path))
            else:
                diffs = [
                    i for i in range(len(actual_seq)) if path_seq[i] != actual_seq[i]
                ]
                if len(diffs) == 1:
                    results.append((diffs[0], path))

        return results

    def summarize(self) -> str:
        """Generate a text summary."""
        lines = [
            f"Decision Space: {len(self.paths)} possible paths",
            f"Total power: {self.total_power:.4f}",
            "",
            "Output Distribution:",
        ]

        for output, power in sorted(self.output_distribution.items()):
            lines.append(f"  {output}: {power:.1%}")

        lines.append("")
        lines.append("All Paths:")

        for i, path in enumerate(self.paths):
            marker = " ← ACTUAL" if path.is_actual else ""
            lines.append(f"  [{i}] {path.signature}")
            lines.append(f"      Power: {path.total_power:.1%}{marker}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich display for Jupyter."""
        rows = ""
        for path in self.paths:
            bg = "background: #e8f5e9;" if path.is_actual else ""
            marker = "✓" if path.is_actual else ""
            rows += f"""
            <tr style="{bg}">
                <td style="padding: 4px;">{marker}</td>
                <td style="padding: 4px; font-family: monospace; font-size: 11px;">{path.signature}</td>
                <td style="padding: 4px; text-align: center;"><b>{path.output_value}</b></td>
                <td style="padding: 4px; text-align: center;">{path.total_power:.1%}</td>
            </tr>
            """

        output_dist = " | ".join(
            f"{v}: {p:.0%}" for v, p in sorted(self.output_distribution.items())
        )

        return f"""
        <div style="border: 1px solid #ccc; padding: 12px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0;">Decision Space ({len(self.paths)} paths)</h4>
            <p><b>Outputs:</b> {output_dist}</p>
            <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 4px; width: 30px;"></th>
                    <th style="padding: 4px; text-align: left;">Path</th>
                    <th style="padding: 4px;">Output</th>
                    <th style="padding: 4px;">Power</th>
                </tr>
                {rows}
            </table>
        </div>
        """


# =============================================================================
# Core Tracked Value
# =============================================================================


@dataclass
class TrackedValue:
    """
    A value that carries contribution tracking information through computation.

    Tracks both:
    - Normalized contributions (backward compatible, weights sum to 1)
    - Signed contributions (SHAP-like, values sum to output)
    - Decision contributions (what drove branching decisions)

    Attributes:
        value: The actual computed numeric value
        contributions: Dict mapping terminal names to normalized weight (0-1)
        signed_contributions: SignedContributions for SHAP-like attribution
        decision_contributions: Dict for decision-level attribution
        path: String representation of computation (for debugging)
        decision_indices: Set of decision indices made during this value's computation
    """

    value: float
    contributions: Dict[str, float] = field(default_factory=dict)
    signed_contributions: SignedContributions = field(
        default_factory=SignedContributions
    )
    decision_contributions: Dict[str, float] = field(default_factory=dict)
    path: str = ""
    decision_indices: set = field(
        default_factory=set
    )  # Track which decisions were made computing this value

    def __repr__(self):
        contrib_str = ", ".join(
            f"{k}: {v:.2%}"
            for k, v in sorted(self.contributions.items(), key=lambda x: -x[1])[:3]
        )
        return f"TrackedValue({self.value:.4f}, [{contrib_str}])"

    def __str__(self):
        """Return string representation suitable for DEAP compilation.

        DEAP's gp.compile() uses str() of terminal values in the generated
        lambda function string. We need to return just the numeric value
        to avoid syntax errors from the detailed repr format.
        """
        return repr(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    @staticmethod
    def from_terminal(name: str, value: float) -> "TrackedValue":
        """Create a TrackedValue from a terminal (leaf node)."""
        return TrackedValue(
            value=value,
            contributions={name: 1.0},
            signed_contributions=SignedContributions(
                contributions={name: value},
                baseline=0.0,
            ),
            decision_contributions={},
            path=name,
        )

    @staticmethod
    def from_constant(value: float, name: str = "_constant") -> "TrackedValue":
        """
        Create a TrackedValue from a constant (ephemeral or fixed).

        Constants do not receive their own explicit contribution tracking.
        Instead, their value is stored entirely in the baseline, so that
        when a constant interacts with feature terminals (e.g. via
        multiplication or addition), the constant's effect is absorbed
        into the feature contributions rather than competing with them.

        Example: For f(x, y) = TWO * x + y where TWO=2, x=3, y=1:
            TWO * x = 6, contributions come entirely from x (amplified by 2)
            + y = 7, x contributes 6/7, y contributes 1/7
            No contribution is attributed to TWO itself.
        """
        path = name if name != "_constant" else str(value)
        return TrackedValue(
            value=value,
            contributions={},  # No normalized contribution — constant is implicit
            signed_contributions=SignedContributions(
                contributions={},
                baseline=value,  # Entire value goes into baseline
            ),
            decision_contributions={},
            path=path,
        )

    def normalize_contributions(self) -> "TrackedValue":
        """Normalize contributions to sum to 1.0."""
        total = sum(self.contributions.values())
        if total > 0:
            self.contributions = {k: v / total for k, v in self.contributions.items()}
        return self

    def get_combined_contributions(self) -> Dict[str, float]:
        """Get combined value + decision contributions (normalized)."""
        combined = self.contributions.copy()
        for k, v in self.decision_contributions.items():
            combined[k] = combined.get(k, 0) + v

        # Re-normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        return combined

    def get_signed_attribution(self) -> Dict[str, float]:
        """Get signed contributions that sum to the output value."""
        return self.signed_contributions.contributions.copy()


# =============================================================================
# Signed Contribution Merging Functions
# =============================================================================


def merge_signed_additive(
    a: TrackedValue, b: TrackedValue, is_sub: bool = False
) -> SignedContributions:
    """
    Merge signed contributions for addition/subtraction.

    For addition: contributions pass through directly
    For subtraction: b's contributions are negated
    """
    result = SignedContributions(
        baseline=a.signed_contributions.baseline
        + (
            b.signed_contributions.baseline
            if not is_sub
            else -b.signed_contributions.baseline
        )
    )

    # Copy a's contributions
    for k, v in a.signed_contributions.contributions.items():
        result.contributions[k] = result.contributions.get(k, 0) + v

    # Add/subtract b's contributions
    sign = -1 if is_sub else 1
    for k, v in b.signed_contributions.contributions.items():
        result.contributions[k] = result.contributions.get(k, 0) + sign * v

    return result


def merge_signed_multiplicative(
    a: TrackedValue, b: TrackedValue
) -> SignedContributions:
    """
    Merge signed contributions for multiplication using ANOVA decomposition.

    x * y = 1 + (x-1) + (y-1) + (x-1)(y-1)

    When one operand is a constant (empty contributions, value in baseline),
    its effect amplifies the other operand's feature contributions rather
    than vanishing. For example, TWO * x where TWO=2, x=3:
        result = 6, and x should get credited for all of it (minus baseline).
    """
    result_value = a.value * b.value

    a_has_contribs = len(a.signed_contributions.contributions) > 0
    b_has_contribs = len(b.signed_contributions.contributions) > 0

    # Special case: one operand is a pure constant (no feature contributions).
    # In this case, the constant acts as a scaling factor — the other
    # operand's contributions are simply scaled by the constant's value.
    if not a_has_contribs and b_has_contribs:
        # a is a constant scaling factor for b's features
        scale = a.value
        return SignedContributions(
            contributions={
                k: v * scale
                for k, v in b.signed_contributions.contributions.items()
            },
            baseline=b.signed_contributions.baseline * scale,
        )

    if a_has_contribs and not b_has_contribs:
        # b is a constant scaling factor for a's features
        scale = b.value
        return SignedContributions(
            contributions={
                k: v * scale
                for k, v in a.signed_contributions.contributions.items()
            },
            baseline=a.signed_contributions.baseline * scale,
        )

    if not a_has_contribs and not b_has_contribs:
        # Both are constants — result is pure baseline
        return SignedContributions(baseline=result_value)

    # General case: both operands have feature contributions.
    # Use ANOVA decomposition: x * y = 1 + (x-1) + (y-1) + (x-1)(y-1)
    a_effect = a.value - 1.0
    b_effect = b.value - 1.0
    interaction = a_effect * b_effect

    result = SignedContributions(baseline=1.0)

    # Distribute a_effect across a's contributors.
    # Each feature's weight is its share of the total feature contribution
    # (i.e., the sum of signed contributions excluding baseline).
    # This ensures the baseline portion of the operand doesn't dilute
    # the distribution of effects to features.
    a_feature_total = sum(a.signed_contributions.contributions.values())
    if abs(a_feature_total) > 1e-15:
        for k, v in a.signed_contributions.contributions.items():
            weight = v / a_feature_total
            result.contributions[k] = result.contributions.get(k, 0) + a_effect * weight
    elif a.value != 0:
        # Features contribute nothing; a_effect comes from baseline
        result.baseline += a_effect
    else:
        result.baseline += a_effect

    # Distribute b_effect across b's contributors
    b_feature_total = sum(b.signed_contributions.contributions.values())
    if abs(b_feature_total) > 1e-15:
        for k, v in b.signed_contributions.contributions.items():
            weight = v / b_feature_total
            result.contributions[k] = result.contributions.get(k, 0) + b_effect * weight
    elif b.value != 0:
        result.baseline += b_effect
    else:
        result.baseline += b_effect

    # Distribute interaction proportionally to all contributors
    all_contribs_abs = {}
    for k, v in a.signed_contributions.contributions.items():
        all_contribs_abs[k] = all_contribs_abs.get(k, 0) + abs(v)
    for k, v in b.signed_contributions.contributions.items():
        all_contribs_abs[k] = all_contribs_abs.get(k, 0) + abs(v)

    total_abs = sum(all_contribs_abs.values())
    if total_abs > 0:
        for k, v in all_contribs_abs.items():
            weight = v / total_abs
            result.contributions[k] = (
                result.contributions.get(k, 0) + interaction * weight
            )
    else:
        result.baseline += interaction

    return result


def merge_signed_passthrough(
    a: TrackedValue, result_value: float
) -> SignedContributions:
    """
    Pass through signed contributions with scaling for unary ops.

    Scales contributions so they sum to the new result value.
    """
    if result_value == 0:
        # Output is zero — all contributions should be zero
        return SignedContributions(baseline=0.0)

    if a.value == 0:
        # Input is zero but output is non-zero (e.g. abs(0) edge case)
        # Can't scale from zero; attribute entirely to baseline
        return SignedContributions(baseline=result_value)

    scale = result_value / a.value

    return SignedContributions(
        contributions={
            k: v * scale for k, v in a.signed_contributions.contributions.items()
        },
        baseline=a.signed_contributions.baseline * scale,
    )


# =============================================================================
# Normalized Contribution Merging (Backward Compatible)
# =============================================================================


def merge_contributions_additive(
    a: TrackedValue, b: TrackedValue, result_value: float
) -> Dict[str, float]:
    """
    Merge normalized contributions for additive operations.

    When one operand has empty contributions (e.g. a constant), its
    magnitude-weight share is redistributed to the other operand's
    features, ensuring contributions still sum to ~1.0.
    """
    total_magnitude = abs(a.value) + abs(b.value)
    if total_magnitude == 0:
        merged = {}
        all_keys = set(a.contributions.keys()) | set(b.contributions.keys())
        for key in all_keys:
            merged[key] = (
                a.contributions.get(key, 0) + b.contributions.get(key, 0)
            ) / 2
        return merged

    a_weight = abs(a.value) / total_magnitude
    b_weight = abs(b.value) / total_magnitude

    merged = {}
    for key, val in a.contributions.items():
        merged[key] = merged.get(key, 0) + val * a_weight
    for key, val in b.contributions.items():
        merged[key] = merged.get(key, 0) + val * b_weight

    # Renormalize: if one side had empty contributions (constant),
    # its weight share was lost. Redistribute so contributions sum to ~1.
    total = sum(merged.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        merged = {k: v / total for k, v in merged.items()}

    return merged


def merge_decision_contributions(a: TrackedValue, b: TrackedValue) -> Dict[str, float]:
    """Merge decision contributions from two TrackedValues."""
    merged = a.decision_contributions.copy()
    for k, v in b.decision_contributions.items():
        merged[k] = merged.get(k, 0) + v

    # Normalize
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}
    return merged


def merge_contributions_multiplicative(
    a: TrackedValue, b: TrackedValue, result_value: float
) -> Dict[str, float]:
    """
    Merge normalized contributions for multiplicative operations.

    When one operand has empty contributions (e.g. a constant), its
    log-weight share is redistributed to the other operand's features.
    """
    if a.value == 0 or b.value == 0:
        if a.value == 0 and b.value == 0:
            return merge_contributions_additive(a, b, result_value)
        elif a.value == 0:
            return a.contributions.copy()
        else:
            return b.contributions.copy()

    log_a = math.log(abs(a.value) + 1e-10)
    log_b = math.log(abs(b.value) + 1e-10)
    total_log = abs(log_a) + abs(log_b)

    if total_log == 0:
        a_weight = b_weight = 0.5
    else:
        a_weight = abs(log_a) / total_log
        b_weight = abs(log_b) / total_log

    merged = {}
    for key, val in a.contributions.items():
        merged[key] = merged.get(key, 0) + val * a_weight
    for key, val in b.contributions.items():
        merged[key] = merged.get(key, 0) + val * b_weight

    # Renormalize if one side had empty contributions (constant)
    total = sum(merged.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        merged = {k: v / total for k, v in merged.items()}

    return merged


def merge_contributions_passthrough(source: TrackedValue) -> Dict[str, float]:
    """Pass through contributions unchanged (for unary operations)."""
    return source.contributions.copy()


# =============================================================================
# Tracked Primitive Operations
# =============================================================================


class TrackedPrimitives:
    """
    Collection of primitive operations that track contributions.

    Each operation returns a TrackedValue with properly attributed contributions
    (both normalized and signed).

    Note: This class uses class-level state for tracking decision points during
    evaluation. This is thread-safe for single-threaded use only.
    """

    # Store decision points and paths during evaluation
    _decisions: List[Decision] = []
    _decision_paths: List[DecisionPath] = []
    _execution_path: List[str] = []
    _current_depth: int = 0

    @classmethod
    def reset_tracking(cls):
        """Reset tracking state before a new evaluation."""
        cls._decisions = []
        cls._decision_paths = []
        cls._execution_path = []
        cls._current_depth = 0

    @classmethod
    def get_decisions(cls) -> List[Decision]:
        return cls._decisions.copy()

    @classmethod
    def get_execution_path(cls) -> List[str]:
        return cls._execution_path.copy()

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def add(a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked addition."""
        result_value = a.value + b.value
        contributions = merge_contributions_additive(a, b, result_value)
        signed = merge_signed_additive(a, b, is_sub=False)
        decision_contribs = merge_decision_contributions(a, b)
        return TrackedValue(
            value=result_value,
            contributions=contributions,
            signed_contributions=signed,
            decision_contributions=decision_contribs,
            path=f"({a.path} + {b.path})",
        )

    @staticmethod
    def sub(a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked subtraction."""
        result_value = a.value - b.value
        contributions = merge_contributions_additive(a, b, result_value)
        signed = merge_signed_additive(a, b, is_sub=True)
        decision_contribs = merge_decision_contributions(a, b)
        return TrackedValue(
            value=result_value,
            contributions=contributions,
            signed_contributions=signed,
            decision_contributions=decision_contribs,
            path=f"({a.path} - {b.path})",
        )

    @staticmethod
    def mul(a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked multiplication."""
        result_value = a.value * b.value
        contributions = merge_contributions_multiplicative(a, b, result_value)
        signed = merge_signed_multiplicative(a, b)
        decision_contribs = merge_decision_contributions(a, b)
        return TrackedValue(
            value=result_value,
            contributions=contributions,
            signed_contributions=signed,
            decision_contributions=decision_contribs,
            path=f"({a.path} * {b.path})",
        )

    @staticmethod
    def div(a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked protected division (returns 1 if divisor is near zero)."""
        if abs(b.value) < 1e-10:
            result_value = 1.0
            contributions = b.contributions.copy()
            signed = SignedContributions(
                contributions=b.signed_contributions.contributions.copy(), baseline=1.0
            )
        else:
            result_value = a.value / b.value
            contributions = merge_contributions_multiplicative(a, b, result_value)

            # Division a/b = a * (1/b).
            # Create a virtual TrackedValue for 1/b that inherits b's
            # contribution structure (negated, since increasing b decreases 1/b).
            recip_value = 1.0 / b.value

            # Scale b's signed contributions to represent 1/b:
            # If b goes from baseline=1 to b.value, then 1/b goes from 1/1=1 to 1/b.value.
            # The passthrough scaling handles this correctly.
            recip_signed = merge_signed_passthrough(b, recip_value)

            # Now merge a * (1/b) using the standard multiplicative merge
            recip_tv = TrackedValue(
                value=recip_value,
                contributions=b.contributions.copy(),
                signed_contributions=recip_signed,
                decision_contributions={},
                path=f"(1/{b.path})",
            )
            signed = merge_signed_multiplicative(a, recip_tv)

        decision_contribs = merge_decision_contributions(a, b)
        return TrackedValue(
            value=result_value,
            contributions=contributions,
            signed_contributions=signed,
            decision_contributions=decision_contribs,
            path=f"({a.path} / {b.path})",
        )

    @staticmethod
    def neg(a: TrackedValue) -> TrackedValue:
        """Tracked negation."""
        signed = SignedContributions(
            contributions={
                k: -v for k, v in a.signed_contributions.contributions.items()
            },
            baseline=-a.signed_contributions.baseline,
        )
        return TrackedValue(
            value=-a.value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"(-{a.path})",
        )

    @staticmethod
    def abs_(a: TrackedValue) -> TrackedValue:
        """Tracked absolute value."""
        if a.value < 0:
            return TrackedPrimitives.neg(a)
        return TrackedValue(
            value=abs(a.value),
            contributions=merge_contributions_passthrough(a),
            signed_contributions=a.signed_contributions.copy(),
            decision_contributions=a.decision_contributions.copy(),
            path=f"|{a.path}|",
        )

    @staticmethod
    def sqrt(a: TrackedValue) -> TrackedValue:
        """Tracked protected square root."""
        result_value = math.sqrt(abs(a.value))
        signed = merge_signed_passthrough(a, result_value)
        return TrackedValue(
            value=result_value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"sqrt({a.path})",
        )

    @staticmethod
    def square(a: TrackedValue) -> TrackedValue:
        """Tracked square."""
        result_value = a.value * a.value

        # If result is zero, all contributions are zero
        if result_value == 0:
            signed = SignedContributions(baseline=0.0)
            return TrackedValue(
                value=0.0,
                contributions=merge_contributions_passthrough(a),
                signed_contributions=signed,
                decision_contributions=a.decision_contributions.copy(),
                path=f"({a.path})²",
            )

        # x² decomposition: 1 + 2(x-1) + (x-1)²
        main_effect = 2 * (a.value - 1.0)
        interaction = (a.value - 1.0) ** 2

        signed = SignedContributions(baseline=1.0)
        a_feature_total = sum(a.signed_contributions.contributions.values())
        if abs(a_feature_total) > 1e-15:
            for k, v in a.signed_contributions.contributions.items():
                weight = v / a_feature_total
                signed.contributions[k] = main_effect * weight
        else:
            signed.baseline += main_effect

        # Distribute interaction
        total_abs = sum(abs(v) for v in a.signed_contributions.contributions.values())
        if total_abs > 0:
            for k, v in a.signed_contributions.contributions.items():
                weight = abs(v) / total_abs
                signed.contributions[k] = (
                    signed.contributions.get(k, 0) + interaction * weight
                )
        else:
            signed.baseline += interaction

        return TrackedValue(
            value=result_value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"({a.path})²",
        )

    @staticmethod
    def pow_(a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked protected power."""
        try:
            if a.value < 0 and not float(b.value).is_integer():
                result_value = 0.0
            else:
                result_value = math.pow(abs(a.value), b.value)
                if math.isnan(result_value) or math.isinf(result_value):
                    result_value = 1.0
        except (ValueError, OverflowError):
            result_value = 1.0

        contributions = merge_contributions_multiplicative(a, b, result_value)
        signed = merge_signed_multiplicative(a, b)  # Approximation
        decision_contribs = merge_decision_contributions(a, b)
        return TrackedValue(
            value=result_value,
            contributions=contributions,
            signed_contributions=signed,
            decision_contributions=decision_contribs,
            path=f"({a.path}^{b.path})",
        )

    @staticmethod
    def sin(a: TrackedValue) -> TrackedValue:
        """Tracked sine."""
        result_value = math.sin(a.value)
        signed = merge_signed_passthrough(a, result_value)
        return TrackedValue(
            value=result_value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"sin({a.path})",
        )

    @staticmethod
    def cos(a: TrackedValue) -> TrackedValue:
        """Tracked cosine."""
        result_value = math.cos(a.value)
        signed = merge_signed_passthrough(a, result_value)
        return TrackedValue(
            value=result_value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"cos({a.path})",
        )

    @staticmethod
    def log_(a: TrackedValue) -> TrackedValue:
        """Tracked protected natural log."""
        if a.value <= 0:
            result_value = 0.0
        else:
            result_value = math.log(a.value)
        signed = merge_signed_passthrough(a, result_value)
        return TrackedValue(
            value=result_value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"log({a.path})",
        )

    @staticmethod
    def exp_(a: TrackedValue) -> TrackedValue:
        """Tracked protected exponential."""
        try:
            result_value = math.exp(min(a.value, 700))
        except OverflowError:
            result_value = float("inf")
        signed = merge_signed_passthrough(a, result_value)
        return TrackedValue(
            value=result_value,
            contributions=merge_contributions_passthrough(a),
            signed_contributions=signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"exp({a.path})",
        )

    @classmethod
    def max_(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """
        Tracked maximum (implicit decision with passthrough).

        Equivalent to: IFGTE(a, b, a, b)

        This is a decision that:
        1. Compares a >= b
        2. Passes through the VALUE and CONTRIBUTIONS of the selected input

        Unlike SIGN/CMP which output fixed values, MAX passes through
        the actual input value, so contributions flow through.
        """
        result = a.value >= b.value

        # Record formal decision
        decision, decision_idx = cls._record_decision(
            ComparisonOp.GTE,
            a,
            b,
            result,
            then_expr=a.path,
            else_expr=b.path,
        )

        if result:
            # a >= b: pass through a's value and contributions
            cls._execution_path.append("left")
            return TrackedValue(
                value=a.value,
                contributions=a.contributions.copy(),
                signed_contributions=a.signed_contributions.copy(),
                decision_contributions=merge_decision_contributions(a, b),
                path=f"max({a.path}, {b.path})",
                decision_indices={decision_idx},
            )
        else:
            # a < b: pass through b's value and contributions
            cls._execution_path.append("right")
            return TrackedValue(
                value=b.value,
                contributions=b.contributions.copy(),
                signed_contributions=b.signed_contributions.copy(),
                decision_contributions=merge_decision_contributions(a, b),
                path=f"max({a.path}, {b.path})",
                decision_indices={decision_idx},
            )

    @classmethod
    def min_(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """
        Tracked minimum (implicit decision with passthrough).

        Equivalent to: IFLTE(a, b, a, b)

        This is a decision that:
        1. Compares a <= b
        2. Passes through the VALUE and CONTRIBUTIONS of the selected input

        Unlike SIGN/CMP which output fixed values, MIN passes through
        the actual input value, so contributions flow through.
        """
        result = a.value <= b.value

        # Record formal decision
        decision, decision_idx = cls._record_decision(
            ComparisonOp.LTE,
            a,
            b,
            result,
            then_expr=a.path,
            else_expr=b.path,
        )

        if result:
            # a <= b: pass through a's value and contributions
            cls._execution_path.append("left")
            return TrackedValue(
                value=a.value,
                contributions=a.contributions.copy(),
                signed_contributions=a.signed_contributions.copy(),
                decision_contributions=merge_decision_contributions(a, b),
                path=f"min({a.path}, {b.path})",
                decision_indices={decision_idx},
            )
        else:
            # a > b: pass through b's value and contributions
            cls._execution_path.append("right")
            return TrackedValue(
                value=b.value,
                contributions=b.contributions.copy(),
                signed_contributions=b.signed_contributions.copy(),
                decision_contributions=merge_decision_contributions(a, b),
                path=f"min({a.path}, {b.path})",
                decision_indices={decision_idx},
            )

    # -------------------------------------------------------------------------
    # Activation/Squashing Functions
    # -------------------------------------------------------------------------

    @staticmethod
    def sigmoid(a: TrackedValue) -> TrackedValue:
        """
        Tracked sigmoid function: 1 / (1 + exp(-x))

        This is a smooth, continuous transformation so contributions
        are scaled by the sigmoid's gradient (derivative).

        Derivative: sig(x) * (1 - sig(x))
        """
        import math

        # Numerically stable sigmoid
        x = a.value
        if x >= 0:
            sig_val = 1 / (1 + math.exp(-min(x, 500)))
        else:
            exp_x = math.exp(max(x, -500))
            sig_val = exp_x / (1 + exp_x)

        # Gradient of sigmoid for contribution scaling
        # sig'(x) = sig(x) * (1 - sig(x))
        gradient = sig_val * (1 - sig_val)

        # Scale contributions by gradient (chain rule)
        # Near 0 or 1, gradient is small -> inputs matter less
        # Near 0.5, gradient is maximum (0.25) -> inputs matter most
        scaled_contribs = {
            k: v * gradient * 4 for k, v in a.contributions.items()
        }  # *4 to normalize max gradient

        # Signed contributions also scaled
        new_signed = SignedContributions(
            contributions={
                k: v * gradient * 4
                for k, v in a.signed_contributions.contributions.items()
            },
            baseline=a.signed_contributions.baseline * gradient * 4,
        )

        return TrackedValue(
            value=sig_val,
            contributions=scaled_contribs,
            signed_contributions=new_signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"sig({a.path})",
        )

    @classmethod
    def relu(cls, a: TrackedValue) -> TrackedValue:
        """
        Tracked ReLU: max(0, x)

        This is a DECISION operation because when x <= 0, the output
        is forced to 0 regardless of the input's value. The input's
        contribution is completely masked.

        When x > 0: contributions pass through unchanged
        When x <= 0: contributions are zeroed (masked decision)
        """
        result = a.value > 0

        # Record this as an implicit decision (comparing to zero)
        zero_threshold = TrackedValue.from_constant(0.0, "0")
        cls._record_decision(
            ComparisonOp.GT,
            a,
            zero_threshold,
            result,
            then_expr=a.path,
            else_expr="0",
        )

        if result:
            # x > 0: pass through with full contributions
            cls._execution_path.append("active")
            return TrackedValue(
                value=a.value,
                contributions=a.contributions.copy(),
                signed_contributions=a.signed_contributions.copy(),
                decision_contributions=a.contributions.copy(),  # Input determined the decision
                path=f"relu({a.path})",
            )
        else:
            # x <= 0: output is 0, contributions are masked
            cls._execution_path.append("masked")
            return TrackedValue(
                value=0.0,
                contributions={},  # No contributions - output is constant
                signed_contributions=SignedContributions(
                    contributions={}, baseline=0.0
                ),
                decision_contributions=a.contributions.copy(),  # But input still made the decision
                path=f"relu({a.path})",
            )

    @classmethod
    def sign(cls, a: TrackedValue) -> TrackedValue:
        """
        Tracked sign function: returns -1 if x < 0, else 1

        This is ALWAYS a decision because the output is always a fixed
        value (±1) regardless of the input magnitude. The input only
        determines WHICH fixed value is output.

        The output value is treated as an implicit constant whose
        contribution is distributed across the features that drove
        the decision. This means when SIGN feeds into arithmetic
        (e.g. x * SIGN(y)), the features that determined the sign
        receive value contribution for the output.
        """
        result = a.value >= 0

        # Record decision
        zero_threshold = TrackedValue.from_constant(0.0, "0")
        cls._record_decision(
            ComparisonOp.GTE,
            a,
            zero_threshold,
            result,
            then_expr="+1",
            else_expr="-1",
        )

        output_val = 1.0 if result else -1.0
        branch_name = "positive" if result else "negative"
        cls._execution_path.append(branch_name)

        # Distribute the output value across the input's contributors
        # as if it were a constant produced by those features.
        # This way, when SIGN feeds into arithmetic, the decision-driving
        # features receive value contribution proportional to their role.
        feature_contribs = {
            k: v for k, v in a.contributions.items()
            if not k.startswith("_") and v > 0
        }
        total_weight = sum(feature_contribs.values())

        if total_weight > 0:
            # Distribute output value across decision-driving features
            signed_contribs = {
                k: output_val * (v / total_weight)
                for k, v in feature_contribs.items()
            }
            norm_contribs = {
                k: v / total_weight for k, v in feature_contribs.items()
            }
        else:
            # Fallback: no trackable features, treat as pure baseline
            signed_contribs = {}
            norm_contribs = {}

        return TrackedValue(
            value=output_val,
            contributions=norm_contribs,
            signed_contributions=SignedContributions(
                contributions=signed_contribs,
                baseline=output_val - sum(signed_contribs.values()),
            ),
            decision_contributions=a.contributions.copy(),
            path=f"sign({a.path})",
        )

    @classmethod
    def compare(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """
        Tracked ternary comparison: returns -1 if a < b, 0 if a == b, 1 if a > b

        This is ALWAYS a decision because the output is one of three fixed
        values regardless of the actual magnitudes of a and b.

        The output value is treated as an implicit constant whose
        contribution is distributed across the features that drove
        the comparison decision.

        E.g. for f(x, y, z) = x * CMP(y, z):
            y > z: output = x * 1  -> x contributes 100% of output value
            y = z: output = x * 0
            y < z: output = x * -1 -> x contributes -100% (signed)
        The CMP node itself carries the decision-drivers (y, z) as value
        contributors to the ±1/0 output, which then interact with x
        through multiplication.
        """
        if a.value < b.value:
            output_val = -1.0
            branch = "less"
            cls._record_decision(
                ComparisonOp.LT, a, b, True, then_expr="-1", else_expr="0 or 1"
            )
        elif a.value > b.value:
            output_val = 1.0
            branch = "greater"
            cls._record_decision(
                ComparisonOp.GT, a, b, True, then_expr="+1", else_expr="0 or -1"
            )
        else:
            output_val = 0.0
            branch = "equal"
            cls._record_decision(
                ComparisonOp.LT, a, b, False, then_expr="-1", else_expr="0 or 1"
            )
            cls._record_decision(
                ComparisonOp.GT, a, b, False, then_expr="+1", else_expr="0 or -1"
            )

        cls._execution_path.append(branch)

        # Merge contributions from both operands for decision tracking
        combined_decision = merge_contributions_additive(a, b, 1.0)

        # Distribute the output value across the decision-driving features.
        # Both a and b's features contributed to this comparison.
        feature_contribs = {
            k: v for k, v in combined_decision.items()
            if not k.startswith("_") and v > 0
        }
        total_weight = sum(feature_contribs.values())

        if total_weight > 0 and output_val != 0.0:
            signed_contribs = {
                k: output_val * (v / total_weight)
                for k, v in feature_contribs.items()
            }
            norm_contribs = {
                k: v / total_weight for k, v in feature_contribs.items()
            }
        else:
            signed_contribs = {}
            norm_contribs = {}

        return TrackedValue(
            value=output_val,
            contributions=norm_contribs,
            signed_contributions=SignedContributions(
                contributions=signed_contribs,
                baseline=output_val - sum(signed_contribs.values()),
            ),
            decision_contributions=combined_decision,
            path=f"cmp({a.path}, {b.path})",
        )

    @staticmethod
    def tanh_(a: TrackedValue) -> TrackedValue:
        """
        Tracked hyperbolic tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        Smooth transformation, output in range (-1, 1).
        Contributions scaled by gradient: 1 - tanh(x)^2
        """
        import math

        x = a.value
        # Numerically stable tanh
        x_clamped = max(-500, min(500, x))
        tanh_val = math.tanh(x_clamped)

        # Gradient: 1 - tanh^2
        gradient = 1 - tanh_val**2

        scaled_contribs = {k: v * gradient for k, v in a.contributions.items()}
        new_signed = SignedContributions(
            contributions={
                k: v * gradient for k, v in a.signed_contributions.contributions.items()
            },
            baseline=a.signed_contributions.baseline * gradient,
        )

        return TrackedValue(
            value=tanh_val,
            contributions=scaled_contribs,
            signed_contributions=new_signed,
            decision_contributions=a.decision_contributions.copy(),
            path=f"tanh({a.path})",
        )

    @classmethod
    def leaky_relu(cls, a: TrackedValue, alpha: float = 0.01) -> TrackedValue:
        """
        Tracked Leaky ReLU: x if x > 0, else alpha * x

        Unlike ReLU, this doesn't completely mask negative inputs,
        but it's still a decision point that affects contribution scaling.
        """
        result = a.value > 0

        zero_threshold = TrackedValue.from_constant(0.0, "0")
        cls._record_decision(
            ComparisonOp.GT,
            a,
            zero_threshold,
            result,
            then_expr=a.path,
            else_expr=f"{alpha}*{a.path}",
        )

        if result:
            cls._execution_path.append("active")
            return TrackedValue(
                value=a.value,
                contributions=a.contributions.copy(),
                signed_contributions=a.signed_contributions.copy(),
                decision_contributions=a.contributions.copy(),
                path=f"lrelu({a.path})",
            )
        else:
            # Negative region: scale by alpha
            cls._execution_path.append("leak")
            scaled_contribs = {k: v * alpha for k, v in a.contributions.items()}
            new_signed = SignedContributions(
                contributions={
                    k: v * alpha
                    for k, v in a.signed_contributions.contributions.items()
                },
                baseline=a.signed_contributions.baseline * alpha,
            )
            return TrackedValue(
                value=a.value * alpha,
                contributions=scaled_contribs,
                signed_contributions=new_signed,
                decision_contributions=a.contributions.copy(),
                path=f"lrelu({a.path})",
            )

    # -------------------------------------------------------------------------
    # Comparison Operations (record Decisions)
    # -------------------------------------------------------------------------

    @classmethod
    def _record_decision(
        cls,
        op: ComparisonOp,
        obs: TrackedValue,
        threshold: TrackedValue,
        result: bool,
        then_expr: str = "",
        else_expr: str = "",
    ) -> Tuple[Decision, int]:
        """Record a decision point for later analysis. Returns (decision, index)."""
        decision = Decision(
            op=op,
            observation_value=obs.value,
            threshold_value=threshold.value,
            observation_source=obs.path,
            threshold_source=threshold.path,
            observation_contributors=obs.contributions.copy(),
            threshold_contributors=threshold.contributions.copy(),
            then_expr=then_expr,
            else_expr=else_expr,
            depth=cls._current_depth,
            branch_taken=result,
        )
        idx = len(cls._decisions)
        cls._decisions.append(decision)
        return decision, idx

    @classmethod
    def less_than(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked less-than comparison. Output value attributed to decision-driving features."""
        result = a.value < b.value
        cls._record_decision(ComparisonOp.LT, a, b, result)

        output_val = 1.0 if result else 0.0
        combined_decision = merge_contributions_additive(a, b, 1.0)

        # Distribute output value across decision-driving features
        feature_contribs = {
            k: v for k, v in combined_decision.items()
            if not k.startswith("_") and v > 0
        }
        total_weight = sum(feature_contribs.values())

        if total_weight > 0 and output_val != 0.0:
            signed_contribs = {
                k: output_val * (v / total_weight)
                for k, v in feature_contribs.items()
            }
            norm_contribs = {k: v / total_weight for k, v in feature_contribs.items()}
        else:
            signed_contribs = {}
            norm_contribs = {}

        return TrackedValue(
            value=output_val,
            contributions=norm_contribs,
            signed_contributions=SignedContributions(
                contributions=signed_contribs,
                baseline=output_val - sum(signed_contribs.values()),
            ),
            decision_contributions=combined_decision,
            path=f"({a.path} < {b.path})",
        )

    @classmethod
    def less_than_or_eq(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked less-than-or-equal comparison. Output value attributed to decision-driving features."""
        result = a.value <= b.value
        cls._record_decision(ComparisonOp.LTE, a, b, result)

        output_val = 1.0 if result else 0.0
        combined_decision = merge_contributions_additive(a, b, 1.0)

        feature_contribs = {
            k: v for k, v in combined_decision.items()
            if not k.startswith("_") and v > 0
        }
        total_weight = sum(feature_contribs.values())

        if total_weight > 0 and output_val != 0.0:
            signed_contribs = {
                k: output_val * (v / total_weight)
                for k, v in feature_contribs.items()
            }
            norm_contribs = {k: v / total_weight for k, v in feature_contribs.items()}
        else:
            signed_contribs = {}
            norm_contribs = {}

        return TrackedValue(
            value=output_val,
            contributions=norm_contribs,
            signed_contributions=SignedContributions(
                contributions=signed_contribs,
                baseline=output_val - sum(signed_contribs.values()),
            ),
            decision_contributions=combined_decision,
            path=f"({a.path} <= {b.path})",
        )

    @classmethod
    def greater_than(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked greater-than comparison. Output value attributed to decision-driving features."""
        result = a.value > b.value
        cls._record_decision(ComparisonOp.GT, a, b, result)

        output_val = 1.0 if result else 0.0
        combined_decision = merge_contributions_additive(a, b, 1.0)

        feature_contribs = {
            k: v for k, v in combined_decision.items()
            if not k.startswith("_") and v > 0
        }
        total_weight = sum(feature_contribs.values())

        if total_weight > 0 and output_val != 0.0:
            signed_contribs = {
                k: output_val * (v / total_weight)
                for k, v in feature_contribs.items()
            }
            norm_contribs = {k: v / total_weight for k, v in feature_contribs.items()}
        else:
            signed_contribs = {}
            norm_contribs = {}

        return TrackedValue(
            value=output_val,
            contributions=norm_contribs,
            signed_contributions=SignedContributions(
                contributions=signed_contribs,
                baseline=output_val - sum(signed_contribs.values()),
            ),
            decision_contributions=combined_decision,
            path=f"({a.path} > {b.path})",
        )

    @classmethod
    def greater_than_or_eq(cls, a: TrackedValue, b: TrackedValue) -> TrackedValue:
        """Tracked greater-than-or-equal comparison. Output value attributed to decision-driving features."""
        result = a.value >= b.value
        cls._record_decision(ComparisonOp.GTE, a, b, result)

        output_val = 1.0 if result else 0.0
        combined_decision = merge_contributions_additive(a, b, 1.0)

        feature_contribs = {
            k: v for k, v in combined_decision.items()
            if not k.startswith("_") and v > 0
        }
        total_weight = sum(feature_contribs.values())

        if total_weight > 0 and output_val != 0.0:
            signed_contribs = {
                k: output_val * (v / total_weight)
                for k, v in feature_contribs.items()
            }
            norm_contribs = {k: v / total_weight for k, v in feature_contribs.items()}
        else:
            signed_contribs = {}
            norm_contribs = {}

        return TrackedValue(
            value=output_val,
            contributions=norm_contribs,
            signed_contributions=SignedContributions(
                contributions=signed_contribs,
                baseline=output_val - sum(signed_contribs.values()),
            ),
            decision_contributions=combined_decision,
            path=f"({a.path} >= {b.path})",
        )

    # -------------------------------------------------------------------------
    # Conditional Operations (with Decision-Centric Tracking)
    # -------------------------------------------------------------------------

    @classmethod
    def if_then_else(
        cls,
        condition: TrackedValue,
        true_branch: TrackedValue,
        false_branch: TrackedValue,
    ) -> TrackedValue:
        """Tracked if-then-else."""
        result = condition.value > 0.5

        if result:
            cls._execution_path.append("then")
            decision_contribs = condition.decision_contributions.copy()
            if not decision_contribs:
                decision_contribs = condition.contributions.copy()

            return TrackedValue(
                value=true_branch.value,
                contributions=true_branch.contributions.copy(),
                signed_contributions=true_branch.signed_contributions.copy(),
                decision_contributions=decision_contribs,
                path=f"if({condition.path}, {true_branch.path}, _)",
            )
        else:
            cls._execution_path.append("else")
            decision_contribs = condition.decision_contributions.copy()
            if not decision_contribs:
                decision_contribs = condition.contributions.copy()

            return TrackedValue(
                value=false_branch.value,
                contributions=false_branch.contributions.copy(),
                signed_contributions=false_branch.signed_contributions.copy(),
                decision_contributions=decision_contribs,
                path=f"if({condition.path}, _, {false_branch.path})",
            )

    @classmethod
    def if_less_than(
        cls,
        a: TrackedValue,
        b: TrackedValue,
        true_branch: TrackedValue,
        false_branch: TrackedValue,
    ) -> TrackedValue:
        """Combined if-less-than primitive with decision tracking."""
        result = a.value < b.value
        decision, decision_idx = cls._record_decision(
            ComparisonOp.LT,
            a,
            b,
            result,
            then_expr=true_branch.path,
            else_expr=false_branch.path,
        )

        # Mark decisions from the not-taken branch as inactive
        if result:
            # Taking TRUE branch, mark FALSE branch decisions as inactive
            for idx in false_branch.decision_indices:
                if idx < len(cls._decisions):
                    cls._decisions[idx].active = False
        else:
            # Taking FALSE branch, mark TRUE branch decisions as inactive
            for idx in true_branch.decision_indices:
                if idx < len(cls._decisions):
                    cls._decisions[idx].active = False

        # Combined decision contributions from observation and threshold
        total_mag = abs(a.value) + abs(b.value)
        if total_mag == 0:
            a_weight = b_weight = 0.5
        else:
            a_weight = abs(a.value) / total_mag
            b_weight = abs(b.value) / total_mag

        combined = {}
        for k, v in a.contributions.items():
            combined[k] = combined.get(k, 0) + v * a_weight
        for k, v in b.contributions.items():
            combined[k] = combined.get(k, 0) + v * b_weight

        if result:
            cls._execution_path.append("then")
            # Collect decision indices from the taken branch plus this decision
            new_indices = true_branch.decision_indices | {decision_idx}
            return TrackedValue(
                value=true_branch.value,
                contributions=true_branch.contributions.copy(),
                signed_contributions=true_branch.signed_contributions.copy(),
                decision_contributions=combined,
                path=f"if({a.path}<{b.path}, {true_branch.path}, _)",
                decision_indices=new_indices,
            )
        else:
            cls._execution_path.append("else")
            # Collect decision indices from the taken branch plus this decision
            new_indices = false_branch.decision_indices | {decision_idx}
            return TrackedValue(
                value=false_branch.value,
                contributions=false_branch.contributions.copy(),
                signed_contributions=false_branch.signed_contributions.copy(),
                decision_contributions=combined,
                path=f"if({a.path}<{b.path}, _, {false_branch.path})",
                decision_indices=new_indices,
            )

    @classmethod
    def if_less_than_or_eq(
        cls,
        a: TrackedValue,
        b: TrackedValue,
        true_branch: TrackedValue,
        false_branch: TrackedValue,
    ) -> TrackedValue:
        """Combined if-less-than-or-equal primitive with decision tracking."""
        result = a.value <= b.value
        decision, decision_idx = cls._record_decision(
            ComparisonOp.LTE,
            a,
            b,
            result,
            then_expr=true_branch.path,
            else_expr=false_branch.path,
        )

        # Mark decisions from the not-taken branch as inactive
        if result:
            # Taking TRUE branch, mark FALSE branch decisions as inactive
            for idx in false_branch.decision_indices:
                if idx < len(cls._decisions):
                    cls._decisions[idx].active = False
        else:
            # Taking FALSE branch, mark TRUE branch decisions as inactive
            for idx in true_branch.decision_indices:
                if idx < len(cls._decisions):
                    cls._decisions[idx].active = False

        # Combined decision contributions
        total_mag = abs(a.value) + abs(b.value)
        if total_mag == 0:
            a_weight = b_weight = 0.5
        else:
            a_weight = abs(a.value) / total_mag
            b_weight = abs(b.value) / total_mag

        combined = {}
        for k, v in a.contributions.items():
            combined[k] = combined.get(k, 0) + v * a_weight
        for k, v in b.contributions.items():
            combined[k] = combined.get(k, 0) + v * b_weight

        if result:
            cls._execution_path.append("then")
            # Collect decision indices from the taken branch plus this decision
            new_indices = true_branch.decision_indices | {decision_idx}
            return TrackedValue(
                value=true_branch.value,
                contributions=true_branch.contributions.copy(),
                signed_contributions=true_branch.signed_contributions.copy(),
                decision_contributions=combined,
                path=f"if({a.path}<={b.path}, {true_branch.path}, _)",
                decision_indices=new_indices,
            )
        else:
            cls._execution_path.append("else")
            # Collect decision indices from the taken branch plus this decision
            new_indices = false_branch.decision_indices | {decision_idx}
            return TrackedValue(
                value=false_branch.value,
                contributions=false_branch.contributions.copy(),
                signed_contributions=false_branch.signed_contributions.copy(),
                decision_contributions=combined,
                path=f"if({a.path}<={b.path}, _, {false_branch.path})",
                decision_indices=new_indices,
            )


# =============================================================================
# Evaluation Result (Enhanced)
# =============================================================================


@dataclass
class EvaluationResult:
    """
    Complete result of evaluating a tree with contribution tracking.

    Attributes:
        output: The final TrackedValue output
        decisions: List of all Decision objects (decision-centric)
        decision_space: Optional DecisionSpace with all possible paths
        terminal_values: The original terminal values that were used
        execution_path: Which branches were taken in IF nodes
    """

    output: TrackedValue
    decisions: List[Decision] = field(default_factory=list)
    decision_space: Optional[DecisionSpace] = None
    terminal_values: Dict[str, float] = field(default_factory=dict)
    execution_path: List[str] = field(default_factory=list)

    # Backward compatibility aliases
    @property
    def decision_points(self) -> List[Decision]:
        """Alias for decisions (backward compatibility)."""
        return self.decisions

    def get_top_value_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N contributing terminals to the OUTPUT VALUE (normalized)."""
        sorted_contribs = sorted(
            self.output.contributions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_contribs[:n]

    def get_top_signed_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N contributing terminals by signed value."""
        return self.output.signed_contributions.top_contributors(n)

    def get_top_decision_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N contributing terminals to the DECISIONS."""
        sorted_contribs = sorted(
            self.output.decision_contributions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_contribs[:n]

    def get_top_combined_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N combined contributors (value + decisions)."""
        combined = self.output.get_combined_contributions()
        sorted_contribs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_contribs[:n]

    @property
    def active_decisions(self) -> List[Decision]:
        """Get only decisions that were on the actual execution path."""
        return [d for d in self.decisions if d.active]

    def get_critical_decision(self) -> Optional[Decision]:
        """Get the active decision closest to flipping (smallest relative distance)."""
        active = self.active_decisions
        if not active:
            return None
        scale_fn = lambda d: max(abs(d.threshold_value), 1.0)
        return min(active, key=lambda d: abs(d.distance_to_flip) / scale_fn(d))

    def explain(self, verbose: bool = True) -> str:
        """Generate a human-readable explanation of the evaluation."""
        lines = []
        lines.append(f"Output: {self.output.value:.6f}")

        # Signed contributions (SHAP-like)
        signed = self.output.signed_contributions.top_contributors(5)
        if signed:
            lines.append(f"\nSigned Contributions (sum to output):")
            for name, contrib in signed:
                if name.startswith("_"):
                    continue
                direction = "↑" if contrib > 0 else "↓"
                lines.append(f"  {name}: {contrib:+.4f} {direction}")
            lines.append(
                f"  Baseline: {self.output.signed_contributions.baseline:+.4f}"
            )
            lines.append(f"  Total: {self.output.signed_contributions.total:.4f}")

        # Normalized value contributions
        value_contribs = self.get_top_value_contributors()
        if value_contribs and value_contribs[0][0] != "_constant":
            lines.append(f"\nNormalized Value Contributors:")
            for name, contrib in value_contribs:
                if name.startswith("_"):
                    continue
                orig_val = self.terminal_values.get(name, "?")
                lines.append(f"  {name}: {contrib:.1%} (value: {orig_val})")

        # Decision contributions
        decision_contribs = self.get_top_decision_contributors()
        if decision_contribs:
            lines.append(f"\nDecision Contributors:")
            for name, contrib in decision_contribs:
                if name.startswith("_"):
                    continue
                orig_val = self.terminal_values.get(name, "?")
                lines.append(f"  {name}: {contrib:.1%} (value: {orig_val})")

        # Decision chain (decision-centric)
        if verbose and self.decisions:
            lines.append(f"\n{'=' * 50}")
            lines.append(f"Decision Chain ({len(self.decisions)} decisions):")
            lines.append(f"{'=' * 50}")
            critical = self.get_critical_decision()
            for i, d in enumerate(self.decisions):
                marker = " ← CRITICAL" if d is critical else ""
                lines.append("")
                lines.append(
                    f"Decision {i+1}: {d.observation_source} {d.op.value} {d.threshold_source}{marker}"
                )
                lines.append(
                    f"  Values: {d.observation_value:.4g} {d.op.value} {d.threshold_value:.4g}"
                )
                lines.append(f"  Result: {'TRUE ✓' if d.result else 'FALSE ✗'}")
                lines.append(
                    f"  Satisfaction: {d.satisfaction:.3f} ({d.satisfaction_description})"
                )
                lines.append(f"  Tipping point: {d.tipping_point:.4g}")
                lines.append(f"  Distance to flip: {d.distance_to_flip:+.4g}")
                lines.append(f"  Decision power: {d.power:.1%}")

        # Decision space summary
        if self.decision_space:
            lines.append("")
            lines.append(self.decision_space.summarize())

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich Jupyter display."""
        # Signed contributions
        signed_html = ""
        for name, contrib in self.output.signed_contributions.top_contributors(5):
            if name.startswith("_"):
                continue
            color = "#4CAF50" if contrib > 0 else "#f44336"
            signed_html += f'<div>{name}: <span style="color: {color};">{contrib:+.4f}</span></div>'

        # Decision chain
        chain_html = ""
        for d in self.decisions:
            sat_color = "#4CAF50" if d.satisfaction > 0 else "#f44336"
            chain_html += f"""
            <div style="border-left: 3px solid {'#4CAF50' if d.result else '#f44336'}; 
                        padding: 8px; margin: 5px 0; background: #f9f9f9;">
                <b>{d.observation_source} {d.op.value} {d.threshold_source}</b><br>
                <span style="color: {'#4CAF50' if d.result else '#f44336'};">
                    {'TRUE ✓' if d.result else 'FALSE ✗'}
                </span>
                <span style="margin-left: 15px; color: {sat_color};">
                    Satisfaction: {d.satisfaction:.3f}
                </span>
                <span style="margin-left: 15px;">
                    Flip: {d.distance_to_flip:+.3g}
                </span>
            </div>
            """

        return f"""
        <div style="border: 2px solid #333; padding: 15px; border-radius: 8px;">
            <h3>Evaluation Result</h3>
            <div style="font-size: 18px; margin-bottom: 15px;">
                <b>Output:</b> {self.output.value:.6f}
            </div>
            <h4>Signed Contributions:</h4>
            {signed_html}
            <div style="margin-top: 5px; color: #666;">
                Baseline: {self.output.signed_contributions.baseline:+.4f}, 
                Total: {self.output.signed_contributions.total:.4f}
            </div>
            <h4>Decision Chain:</h4>
            {chain_html if chain_html else '<i>No decisions</i>'}
            {self.decision_space._repr_html_() if self.decision_space else ''}
        </div>
        """


# =============================================================================
# DEAP Integration: Primitive Set Builder
# =============================================================================


def create_tracked_pset(
    name: str,
    input_names: List[str],
    include_trig: bool = False,
    include_exp_log: bool = False,
    include_comparisons: bool = True,
    include_conditionals: bool = True,
    include_activations: bool = True,
) -> gp.PrimitiveSet:
    """
    Create a DEAP PrimitiveSet configured for tracked evaluation.

    Args:
        name: Name for the primitive set
        input_names: Names of input terminals (e.g., ['x', 'y', 'z'])
        include_trig: Include sin/cos
        include_exp_log: Include exp/log
        include_comparisons: Include comparison primitives
        include_conditionals: Include if-then-else primitives
        include_activations: Include activation functions (sigmoid, relu, sign, etc.)

    Returns:
        A configured PrimitiveSet where all operations work with TrackedValue
    """
    pset = gp.PrimitiveSet(name, len(input_names))

    # Rename arguments
    rename_map = {f"ARG{i}": name for i, name in enumerate(input_names)}
    pset.renameArguments(**rename_map)

    # Basic arithmetic
    pset.addPrimitive(TrackedPrimitives.add, 2, name="ADD")
    pset.addPrimitive(TrackedPrimitives.sub, 2, name="SUB")
    pset.addPrimitive(TrackedPrimitives.mul, 2, name="MUL")
    pset.addPrimitive(TrackedPrimitives.div, 2, name="DIV")
    pset.addPrimitive(TrackedPrimitives.neg, 1, name="NEG")
    pset.addPrimitive(TrackedPrimitives.abs_, 1, name="ABS")
    pset.addPrimitive(TrackedPrimitives.sqrt, 1, name="SQRT")
    pset.addPrimitive(TrackedPrimitives.square, 1, name="SQUARE")
    pset.addPrimitive(TrackedPrimitives.max_, 2, name="MAX")
    pset.addPrimitive(TrackedPrimitives.min_, 2, name="MIN")

    if include_trig:
        pset.addPrimitive(TrackedPrimitives.sin, 1, name="SIN")
        pset.addPrimitive(TrackedPrimitives.cos, 1, name="COS")

    if include_exp_log:
        pset.addPrimitive(TrackedPrimitives.exp_, 1, name="EXP")
        pset.addPrimitive(TrackedPrimitives.log_, 1, name="LOG")
        pset.addPrimitive(TrackedPrimitives.pow_, 2, name="POW")

    if include_comparisons:
        pset.addPrimitive(TrackedPrimitives.less_than, 2, name="LT")
        pset.addPrimitive(TrackedPrimitives.less_than_or_eq, 2, name="LTE")
        pset.addPrimitive(TrackedPrimitives.greater_than, 2, name="GT")
        pset.addPrimitive(TrackedPrimitives.greater_than_or_eq, 2, name="GTE")

    if include_conditionals:
        pset.addPrimitive(TrackedPrimitives.if_then_else, 3, name="IF")
        pset.addPrimitive(TrackedPrimitives.if_less_than, 4, name="IFLT")
        pset.addPrimitive(TrackedPrimitives.if_less_than_or_eq, 4, name="IFLTE")

    if include_activations:
        # Smooth activation functions (value contribution via gradient)
        pset.addPrimitive(TrackedPrimitives.sigmoid, 1, name="SIG")
        pset.addPrimitive(TrackedPrimitives.tanh_, 1, name="TANH")

        # Decision-based activations (implicit comparisons to zero)
        pset.addPrimitive(TrackedPrimitives.relu, 1, name="RELU")
        pset.addPrimitive(TrackedPrimitives.sign, 1, name="SIGN")
        pset.addPrimitive(TrackedPrimitives.compare, 2, name="CMP")

        # Leaky ReLU with default alpha
        pset.addPrimitive(
            lambda a: TrackedPrimitives.leaky_relu(a, 0.01), 1, name="LRELU"
        )

    return pset


# =============================================================================
# Evaluation Function
# =============================================================================


def evaluate_with_tracking(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
) -> EvaluationResult:
    """
    Evaluate a GP tree with full contribution tracking.

    Args:
        individual: The GP tree to evaluate
        pset: The primitive set used to create the tree
        terminal_values: Dict mapping terminal names to their values

    Returns:
        EvaluationResult with output, contributions, decisions, etc.
    """
    # Reset tracking state
    TrackedPrimitives.reset_tracking()

    # Convert terminal values to TrackedValues
    tracked_terminals = {
        name: TrackedValue.from_terminal(name, value)
        for name, value in terminal_values.items()
    }

    # Compile and evaluate the tree
    func = gp.compile(individual, pset)

    # Get argument names from pset in the correct order
    # pset.arguments contains the ordered list of argument names
    if hasattr(pset, "arguments"):
        arg_names = pset.arguments
    elif hasattr(pset, "args"):
        arg_names = list(pset.args)
    else:
        arg_names = list(terminal_values.keys())

    args = [tracked_terminals[name] for name in arg_names if name in tracked_terminals]

    # Evaluate
    output = func(*args)

    # Ensure output is a TrackedValue
    if not isinstance(output, TrackedValue):
        output = TrackedValue.from_constant(float(output))

    return EvaluationResult(
        output=output,
        decisions=TrackedPrimitives.get_decisions(),
        terminal_values=terminal_values.copy(),
        execution_path=TrackedPrimitives.get_execution_path(),
    )


# =============================================================================
# Counterfactual Analysis
# =============================================================================


def compute_flip_sensitivity(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
    target_decision_index: int = 0,
    epsilon: float = 0.001,
    max_delta: float = 100.0,
) -> Dict[str, float]:
    """
    Compute how much each terminal would need to change to flip a decision.

    Uses numerical differentiation to estimate sensitivities.
    """
    baseline_result = evaluate_with_tracking(individual, pset, terminal_values)

    if target_decision_index >= len(baseline_result.decisions):
        return {}

    baseline_d = baseline_result.decisions[target_decision_index]
    margin = baseline_d.raw_margin

    sensitivities = {}

    for terminal_name in terminal_values.keys():
        if terminal_name.startswith("_"):
            continue

        perturbed_values = terminal_values.copy()
        perturbed_values[terminal_name] += epsilon

        perturbed_result = evaluate_with_tracking(individual, pset, perturbed_values)

        if target_decision_index >= len(perturbed_result.decisions):
            continue

        perturbed_d = perturbed_result.decisions[target_decision_index]
        gradient = (perturbed_d.raw_margin - margin) / epsilon

        if abs(gradient) > 1e-10:
            delta = -margin / gradient
            delta = max(-max_delta, min(max_delta, delta))
            sensitivities[terminal_name] = delta
        else:
            sensitivities[terminal_name] = float("inf")

    return sensitivities


def find_counterfactual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
    terminals_to_modify: Optional[List[str]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[Dict[str, float], EvaluationResult, bool]:
    """
    Find a counterfactual: minimal changes to flip the critical decision.
    """
    current_values = terminal_values.copy()

    if terminals_to_modify is None:
        terminals_to_modify = [
            k for k in terminal_values.keys() if not k.startswith("_")
        ]

    initial_result = evaluate_with_tracking(individual, pset, terminal_values)
    if not initial_result.decisions:
        return terminal_values, initial_result, False

    initial_d = initial_result.get_critical_decision()
    initial_condition = initial_d.result

    for iteration in range(max_iterations):
        result = evaluate_with_tracking(individual, pset, current_values)

        if not result.decisions:
            break

        d = result.get_critical_decision()

        if d.result != initial_condition:
            return current_values, result, True

        sensitivities = compute_flip_sensitivity(
            individual,
            pset,
            current_values,
            target_decision_index=result.decisions.index(d),
        )

        best_terminal = None
        best_delta = float("inf")

        for terminal_name in terminals_to_modify:
            if terminal_name in sensitivities:
                delta = sensitivities[terminal_name]
                if abs(delta) < abs(best_delta):
                    best_terminal = terminal_name
                    best_delta = delta

        if best_terminal is None or abs(best_delta) >= float("inf"):
            break

        overshoot = 1.01
        step = best_delta * overshoot
        current_values[best_terminal] += step

    final_result = evaluate_with_tracking(individual, pset, current_values)
    if final_result.decisions:
        final_d = final_result.get_critical_decision()
        success = final_d.result != initial_condition
    else:
        success = False

    return current_values, final_result, success


# =============================================================================
# Convenience Functions
# =============================================================================


def explain_prediction(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    terminal_values: Dict[str, float],
    include_counterfactual: bool = True,
    verbose: bool = True,
) -> str:
    """
    Generate a complete explanation for a single prediction.
    """
    result = evaluate_with_tracking(individual, pset, terminal_values)

    lines = [result.explain(verbose=verbose)]

    if include_counterfactual and result.decisions:
        lines.append("\n" + "=" * 50)
        lines.append("Counterfactual Analysis:")

        sensitivities = compute_flip_sensitivity(
            individual, pset, terminal_values, target_decision_index=0
        )

        if sensitivities:
            valid_changes = [
                (abs(v), k, v)
                for k, v in sensitivities.items()
                if abs(v) < float("inf")
            ]

            if valid_changes:
                min_change = min(valid_changes)
                lines.append(
                    f"  Easiest flip: Change {min_change[1]} by {min_change[2]:+.4f}"
                )
                lines.append(
                    f"    (from {terminal_values[min_change[1]]:.4f} "
                    f"to {terminal_values[min_change[1]] + min_change[2]:.4f})"
                )

            lines.append("\n  All sensitivities (delta needed to flip):")
            for name, delta in sorted(sensitivities.items(), key=lambda x: abs(x[1])):
                if abs(delta) < float("inf"):
                    lines.append(f"    {name}: {delta:+.4f}")

    return "\n".join(lines)


# =============================================================================
# Explanation Synthesis
# =============================================================================


class ExplanationMode(Enum):
    """How the explanation should be framed based on tree structure."""

    VALUE_DOMINANT = "value_dominant"  # Pure arithmetic, no decisions
    DECISION_DOMINANT = "decision_dominant"  # Output determined primarily by branching
    MIXED = "mixed"  # Both value and decision contributions matter


@dataclass
class Factor:
    """A single factor contributing to the prediction."""

    name: str
    importance: float  # Normalized importance (0-1)
    signed_contribution: float  # Actual signed contribution to output
    direction: str  # "increases", "decreases", or "neutral"
    factor_type: str  # "input", "constant", "decision", "interaction"

    def describe(self) -> str:
        """Human-readable description of this factor's influence."""
        if self.factor_type == "decision":
            return f"{self.name} (decision)"

        if abs(self.signed_contribution) < 1e-6:
            return f"{self.name} had negligible effect"

        direction = "increases" if self.signed_contribution > 0 else "decreases"
        return f"{self.name} {direction} output by {abs(self.signed_contribution):.4g}"


@dataclass
class Counterfactual:
    """A minimal change that would produce a different output."""

    feature: str
    current_value: float
    required_value: float
    change_magnitude: float
    change_type: str  # "flip_decision", "change_output"
    resulting_output: Optional[float] = None
    description: str = ""

    @property
    def relative_change(self) -> float:
        """Change as a proportion of current value."""
        if abs(self.current_value) < 1e-9:
            return float("inf") if self.change_magnitude != 0 else 0
        return abs(self.change_magnitude / self.current_value)


@dataclass
class Explanation:
    """
    Unified explanation answering WHY, HOW CONFIDENT, and WHAT TO CHANGE.

    This synthesizes the various FCP metrics into a coherent explanation
    that adapts based on the tree structure (value vs decision dominant).
    """

    # Metadata
    output_value: float
    mode: ExplanationMode
    is_classification: bool

    # WHY - What factors drove the output?
    primary_factors: List[Factor]
    secondary_factors: List[Factor]

    # HOW CONFIDENT - How stable is the prediction?
    stability_score: float  # 0-1, higher = more stable
    critical_decision: Optional[Decision]
    critical_factor: Optional[str]
    margin_to_change: Optional[float]

    # WHAT TO CHANGE - Counterfactuals
    counterfactuals: List[Counterfactual]
    easiest_counterfactual: Optional[Counterfactual]

    # Raw data for further analysis
    raw_result: Optional[Any] = None

    def _generate_why_narrative(self) -> str:
        """Generate narrative explaining WHY the output occurred."""
        lines = []

        if self.is_classification:
            class_label = (
                "TRUE (1)"
                if self.output_value == 1.0
                else "FALSE (0)" if self.output_value == 0.0 else str(self.output_value)
            )
            lines.append(f"The model predicted {class_label}.")
        else:
            lines.append(f"The model produced an output of {self.output_value:.4g}.")

        if not self.primary_factors:
            lines.append("No significant contributing factors were identified.")
            return " ".join(lines)

        # Describe primary factors
        if self.mode == ExplanationMode.DECISION_DOMINANT:
            lines.append("This was determined by the following decision(s):")
            for f in self.primary_factors[:3]:
                lines.append(f"  • {f.describe()}")
        elif self.mode == ExplanationMode.VALUE_DOMINANT:
            lines.append("The main contributing factors were:")
            pos_factors = [f for f in self.primary_factors if f.signed_contribution > 0]
            neg_factors = [f for f in self.primary_factors if f.signed_contribution < 0]

            if pos_factors:
                pos_desc = ", ".join(
                    f"{f.name} (+{f.signed_contribution:.4g})" for f in pos_factors[:3]
                )
                lines.append(f"  • Increasing output: {pos_desc}")
            if neg_factors:
                neg_desc = ", ".join(
                    f"{f.name} ({f.signed_contribution:.4g})" for f in neg_factors[:3]
                )
                lines.append(f"  • Decreasing output: {neg_desc}")
        else:  # MIXED
            lines.append("This was determined by a combination of:")
            decision_factors = [
                f for f in self.primary_factors if f.factor_type == "decision"
            ]
            value_factors = [
                f for f in self.primary_factors if f.factor_type != "decision"
            ]

            if decision_factors:
                lines.append(
                    f"  • Decisions: {', '.join(f.name for f in decision_factors[:2])}"
                )
            if value_factors:
                lines.append(
                    f"  • Values: {', '.join(f.name for f in value_factors[:3])}"
                )

        return "\n".join(lines)

    def _generate_confidence_narrative(self) -> str:
        """Generate narrative about prediction stability/confidence."""
        lines = []

        # Interpret stability score
        if self.stability_score > 0.8:
            stability_desc = "highly stable"
        elif self.stability_score > 0.5:
            stability_desc = "moderately stable"
        elif self.stability_score > 0.2:
            stability_desc = "somewhat unstable"
        else:
            stability_desc = "very unstable (close to decision boundary)"

        lines.append(
            f"The prediction is {stability_desc} (stability: {self.stability_score:.0%})."
        )

        if self.critical_decision:
            d = self.critical_decision
            lines.append(
                f"The most critical decision was '{d.observation_source} {d.op.value} {d.threshold_source}' "
                f"with a margin of {abs(d.distance_to_flip):.4g}."
            )
            if abs(d.distance_to_flip) < abs(d.threshold_value) * 0.1:
                lines.append(
                    "This decision was barely satisfied - small changes could flip it."
                )
        elif self.critical_factor and self.margin_to_change is not None:
            lines.append(
                f"The output is most sensitive to changes in '{self.critical_factor}' "
                f"(change of {self.margin_to_change:+.4g} would significantly alter the result)."
            )

        return "\n".join(lines)

    def _generate_counterfactual_narrative(self) -> str:
        """Generate narrative about what changes would alter the prediction."""
        lines = []

        if not self.counterfactuals:
            lines.append("No simple counterfactuals were identified.")
            return "\n".join(lines)

        if self.easiest_counterfactual:
            cf = self.easiest_counterfactual
            lines.append("To change the prediction:")

            if cf.change_type == "flip_decision":
                lines.append(
                    f"  • Change '{cf.feature}' from {cf.current_value:.4g} to {cf.required_value:.4g} "
                    f"(Δ = {cf.change_magnitude:+.4g})"
                )
                lines.append(f"    This would flip a critical decision.")
            else:
                lines.append(
                    f"  • Adjust '{cf.feature}' by {cf.change_magnitude:+.4g} "
                    f"(from {cf.current_value:.4g} to {cf.required_value:.4g})"
                )

            if cf.relative_change < float("inf"):
                lines.append(f"    This represents a {cf.relative_change:.1%} change.")

        # Mention other options
        other_cfs = [
            cf for cf in self.counterfactuals if cf != self.easiest_counterfactual
        ]
        if other_cfs:
            other_features = ", ".join(cf.feature for cf in other_cfs[:3])
            lines.append(f"Alternative changes could involve: {other_features}")

        return "\n".join(lines)

    def narrative(self) -> str:
        """Generate complete human-readable explanation."""
        sections = [
            "## Why this prediction?",
            self._generate_why_narrative(),
            "",
            "## How confident is the model?",
            self._generate_confidence_narrative(),
            "",
            "## What would change the prediction?",
            self._generate_counterfactual_narrative(),
        ]
        return "\n".join(sections)

    def summary(self) -> str:
        """Generate a brief one-paragraph summary."""
        parts = []

        # Output
        if self.is_classification:
            class_label = "positive" if self.output_value == 1.0 else "negative"
            parts.append(f"The model predicted {class_label}")
        else:
            parts.append(f"The model output {self.output_value:.4g}")

        # Primary factor
        if self.primary_factors:
            top = self.primary_factors[0]
            parts.append(f"primarily driven by {top.name}")

        # Stability
        if self.stability_score < 0.3:
            parts.append("though this prediction is unstable")
        elif self.stability_score > 0.7:
            parts.append("with high confidence")

        # Counterfactual hint
        if self.easiest_counterfactual:
            cf = self.easiest_counterfactual
            parts.append(
                f"(changing {cf.feature} by {cf.change_magnitude:+.4g} would alter the result)"
            )

        return ", ".join(parts) + "."

    def _repr_html_(self) -> str:
        """Rich display for Jupyter."""
        # Stability color
        if self.stability_score > 0.7:
            stab_color = "#22c55e"
        elif self.stability_score > 0.3:
            stab_color = "#f59e0b"
        else:
            stab_color = "#ef4444"

        # Primary factors HTML
        factors_html = ""
        for f in self.primary_factors[:5]:
            bar_color = "#22c55e" if f.signed_contribution > 0 else "#ef4444"
            width = int(abs(f.importance) * 150)
            factors_html += f"""
            <div style="margin: 4px 0; display: flex; align-items: center; gap: 10px;">
                <span style="width: 80px; font-weight: 500;">{f.name}</span>
                <div style="width: {width}px; height: 12px; background: {bar_color}; border-radius: 2px;"></div>
                <span style="color: {bar_color};">{f.signed_contribution:+.4g}</span>
            </div>
            """

        # Counterfactual HTML
        cf_html = ""
        if self.easiest_counterfactual:
            cf = self.easiest_counterfactual
            cf_html = f"""
            <div style="background: #fef3c7; padding: 10px; border-radius: 6px; margin-top: 10px;">
                <b>Easiest change:</b> Set <code>{cf.feature}</code> to {cf.required_value:.4g}
                <span style="color: #92400e;">(Δ = {cf.change_magnitude:+.4g})</span>
            </div>
            """

        return f"""
        <div style="border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin: 10px 0;
                    background: #ffffff; font-family: -apple-system, sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #1f2937;">Prediction Explanation</h3>
                <span style="background: {stab_color}; color: white; padding: 4px 10px; 
                             border-radius: 12px; font-size: 12px;">
                    Stability: {self.stability_score:.0%}
                </span>
            </div>
            
            <div style="background: #f9fafb; padding: 10px; border-radius: 6px; margin-bottom: 12px;">
                <b>Output:</b> <span style="font-size: 18px;">{self.output_value:.4g}</span>
                <span style="color: #6b7280; margin-left: 10px;">({self.mode.value})</span>
            </div>
            
            <div style="margin-bottom: 12px;">
                <b style="color: #374151;">Primary Factors:</b>
                {factors_html if factors_html else '<i style="color: #9ca3af;">None identified</i>'}
            </div>
            
            {cf_html}
        </div>
        """


def synthesize_explanation(
    result: EvaluationResult,
    terminal_values: Optional[Dict[str, float]] = None,
    classification_threshold: float = 0.5,
    actionable_features: Optional[List[str]] = None,
    frozen_features: Optional[List[str]] = None,
    constant_names: Optional[List[str]] = None,
    tree: Optional[Any] = None,
    pset: Optional[Any] = None,
) -> Explanation:
    """
    Synthesize a coherent explanation from an evaluation result.

    This is the main entry point for explanation generation. It analyzes
    the evaluation result and produces a unified explanation that adapts
    based on the tree structure.

    Args:
        result: EvaluationResult from evaluate_with_tracking
        terminal_values: Original input values (for counterfactual calculation)
        classification_threshold: Threshold for binary classification
        actionable_features: If provided, ONLY these features can be suggested
                            for counterfactuals. If None, all non-constant features
                            are considered actionable.
        frozen_features: Features that should never be suggested for counterfactuals
                        (e.g., age, protected characteristics). Applied after
                        actionable_features filter.
        constant_names: Names of constants/model parameters that should never be
                       suggested for change. If None, attempts to auto-detect
                       (names starting with uppercase, containing 'WEIGHT', 'THRESHOLD', etc.)
        tree: The GP tree (required for accurate counterfactual calculation via
              numerical differentiation). If None, counterfactuals may be limited.
        pset: The primitive set (required with tree for counterfactuals).

    Returns:
        Explanation object with structured and narrative explanations
    """
    output = result.output
    decisions = (
        result.active_decisions
        if hasattr(result, "active_decisions")
        else result.decisions
    )

    # Determine if this is classification
    is_classification = output.value in (0.0, 1.0) and len(decisions) > 0

    # Determine explanation mode
    mode = _determine_mode(result)

    # Extract primary and secondary factors
    primary_factors, secondary_factors = _extract_factors(result, mode)

    # Calculate stability
    stability_score, critical_decision, critical_factor, margin = _calculate_stability(
        result, decisions, terminal_values
    )

    # Find counterfactuals (with actionability constraints)
    counterfactuals = _find_counterfactuals(
        result,
        terminal_values,
        decisions,
        actionable_features=actionable_features,
        frozen_features=frozen_features,
        constant_names=constant_names,
        tree=tree,
        pset=pset,
    )
    easiest_cf = (
        min(counterfactuals, key=lambda cf: abs(cf.change_magnitude))
        if counterfactuals
        else None
    )

    return Explanation(
        output_value=output.value,
        mode=mode,
        is_classification=is_classification,
        primary_factors=primary_factors,
        secondary_factors=secondary_factors,
        stability_score=stability_score,
        critical_decision=critical_decision,
        critical_factor=critical_factor,
        margin_to_change=margin,
        counterfactuals=counterfactuals,
        easiest_counterfactual=easiest_cf,
        raw_result=result,
    )


def _determine_mode(result: EvaluationResult) -> ExplanationMode:
    """Determine whether explanation should focus on values, decisions, or both."""
    decisions = (
        result.active_decisions
        if hasattr(result, "active_decisions")
        else result.decisions
    )
    output = result.output

    # No decisions = pure value
    if not decisions:
        return ExplanationMode.VALUE_DOMINANT

    # Check if output is purely decision-determined (0 or 1)
    if output.value in (0.0, 1.0):
        # Check if there are meaningful value contributions beyond the constant
        non_const_contribs = {
            k: v
            for k, v in output.signed_contributions.contributions.items()
            if not k.startswith("_") and abs(v) > 0.01
        }
        if len(non_const_contribs) <= 1:
            return ExplanationMode.DECISION_DOMINANT

    # Check relative importance of decisions vs values
    total_decision_contrib = sum(output.decision_contributions.values())
    total_value_contrib = sum(
        abs(v) for v in output.signed_contributions.contributions.values()
    )

    if total_value_contrib < 0.01:
        return ExplanationMode.DECISION_DOMINANT

    if not decisions or total_decision_contrib < 0.1 * total_value_contrib:
        return ExplanationMode.VALUE_DOMINANT

    return ExplanationMode.MIXED


def _extract_factors(
    result: EvaluationResult,
    mode: ExplanationMode,
) -> Tuple[List[Factor], List[Factor]]:
    """Extract and rank contributing factors."""
    output = result.output
    decisions = (
        result.active_decisions
        if hasattr(result, "active_decisions")
        else result.decisions
    )

    factors = []

    # Value-based factors from signed contributions
    signed = output.signed_contributions
    total_magnitude = sum(
        abs(v) for k, v in signed.contributions.items() if not k.startswith("_")
    )

    for name, contrib in signed.contributions.items():
        if name.startswith("_"):
            continue

        importance = abs(contrib) / total_magnitude if total_magnitude > 0 else 0
        direction = (
            "increases" if contrib > 0 else "decreases" if contrib < 0 else "neutral"
        )

        # Determine factor type
        if name.startswith("w") or name.isupper():
            factor_type = "constant"
        else:
            factor_type = "input"

        factors.append(
            Factor(
                name=name,
                importance=importance,
                signed_contribution=contrib,
                direction=direction,
                factor_type=factor_type,
            )
        )

    # Decision-based factors
    for i, d in enumerate(decisions):
        # Create factor for the decision itself
        decision_name = f"{d.observation_source} {d.op.value} {d.threshold_source}"

        # Importance based on how close to boundary
        margin = abs(d.distance_to_flip)
        scale = max(abs(d.threshold_value), 1.0)
        decision_importance = 1.0 / (
            1.0 + margin / scale
        )  # Closer to boundary = more important

        factors.append(
            Factor(
                name=decision_name,
                importance=decision_importance * d.power,
                signed_contribution=d.power if d.result else -d.power,
                direction="satisfied" if d.result else "not satisfied",
                factor_type="decision",
            )
        )

    # Sort by importance
    factors.sort(key=lambda f: f.importance, reverse=True)

    # Split into primary (top factors) and secondary
    if mode == ExplanationMode.DECISION_DOMINANT:
        # Prioritize decision factors
        decision_factors = [f for f in factors if f.factor_type == "decision"]
        value_factors = [f for f in factors if f.factor_type != "decision"]
        primary = decision_factors[:3] + value_factors[:2]
        secondary = decision_factors[3:] + value_factors[2:]
    elif mode == ExplanationMode.VALUE_DOMINANT:
        # Prioritize value factors
        value_factors = [f for f in factors if f.factor_type != "decision"]
        decision_factors = [f for f in factors if f.factor_type == "decision"]
        primary = value_factors[:5]
        secondary = value_factors[5:] + decision_factors
    else:
        # Mixed - take top of each
        primary = factors[:5]
        secondary = factors[5:]

    return primary, secondary


def _calculate_stability(
    result: EvaluationResult,
    decisions: List[Decision],
    terminal_values: Optional[Dict[str, float]],
) -> Tuple[float, Optional[Decision], Optional[str], Optional[float]]:
    """
    Calculate prediction stability.

    Returns: (stability_score, critical_decision, critical_factor, margin_to_change)
    """
    if not decisions:
        # For pure value models, stability is based on output magnitude
        # (harder to interpret without a threshold)
        return 0.5, None, None, None

    # Find the decision closest to flipping
    critical = result.get_critical_decision()

    if critical:
        # Stability based on distance to flip relative to threshold scale
        margin = abs(critical.distance_to_flip)
        scale = max(abs(critical.threshold_value), 1.0)

        # Map to 0-1 score (further from boundary = more stable)
        # Using sigmoid-like mapping
        relative_margin = margin / scale
        stability = relative_margin / (1 + relative_margin)

        critical_factor = critical.observation_source

        return stability, critical, critical_factor, critical.distance_to_flip

    return 0.5, None, None, None


def _is_likely_constant(name: str) -> bool:
    """
    Heuristic to detect if a terminal name is likely a constant/model parameter.

    Constants typically:
    - Are all uppercase (THRESHOLD, WEIGHT, ONE, ZERO)
    - Start with common prefixes (w1, w2 for weights)
    - Contain keywords like WEIGHT, THRESHOLD, BIAS, CONST
    """
    upper_name = name.upper()

    # All uppercase names are likely constants
    if name.isupper() and len(name) > 1:
        return True

    # Common constant patterns
    constant_patterns = [
        "WEIGHT",
        "THRESHOLD",
        "BIAS",
        "CONST",
        "COEF",
        "BASE",
        "SCALE",
        "FACTOR",
        "PARAM",
        "LIMIT",
        "ONE",
        "ZERO",
        "TRUE",
        "FALSE",
        "MIN",
        "MAX",
    ]
    for pattern in constant_patterns:
        if pattern in upper_name:
            return True

    # Weight-like names: w1, w2, etc. or weight_1, etc.
    if name.startswith("w") and len(name) <= 3 and name[1:].isdigit():
        return True

    return False


def _find_counterfactuals(
    result: EvaluationResult,
    terminal_values: Optional[Dict[str, float]],
    decisions: List[Decision],
    actionable_features: Optional[List[str]] = None,
    frozen_features: Optional[List[str]] = None,
    constant_names: Optional[List[str]] = None,
    tree: Optional[Any] = None,
    pset: Optional[Any] = None,
) -> List[Counterfactual]:
    """
    Find minimal changes that would alter the prediction.

    Uses numerical differentiation (compute_flip_sensitivity) when tree and pset
    are provided for accurate counterfactual calculation even when decisions
    involve computed values (not raw terminals).

    Args:
        result: The evaluation result
        terminal_values: Current values of all terminals
        decisions: Active decisions in the tree
        actionable_features: If provided, ONLY these features can be changed
        frozen_features: Features that cannot be changed (applied after actionable filter)
        constant_names: Known constant names. If None, auto-detects using heuristics.
        tree: The GP tree (enables numerical differentiation for accurate counterfactuals)
        pset: The primitive set (required with tree)

    Returns:
        List of possible counterfactuals, filtered by actionability
    """
    counterfactuals = []

    if not terminal_values or not decisions:
        return counterfactuals

    # Build set of features that can be changed
    if constant_names is not None:
        constants = set(constant_names)
    else:
        # Auto-detect constants
        constants = {
            name for name in terminal_values.keys() if _is_likely_constant(name)
        }

    # Determine which features are actionable
    if actionable_features is not None:
        actionable = set(actionable_features)
    else:
        # All non-constants are potentially actionable
        actionable = set(terminal_values.keys()) - constants

    # Remove frozen features
    if frozen_features:
        actionable -= set(frozen_features)

    # If we have tree and pset, use numerical differentiation for accurate counterfactuals
    if tree is not None and pset is not None:
        # Find the critical decision
        critical = result.get_critical_decision()
        if critical and critical in decisions:
            critical_idx = decisions.index(critical)

            # Compute sensitivities for all terminals
            # Use high max_delta to get real values, not clamped ones
            sensitivities = compute_flip_sensitivity(
                tree,
                pset,
                terminal_values,
                target_decision_index=critical_idx,
                max_delta=1e12,  # Effectively no clamp
            )

            # Create counterfactuals for actionable features
            for feature, delta in sensitivities.items():
                if feature not in actionable:
                    continue

                if abs(delta) >= float("inf") or abs(delta) > 1e10:
                    continue  # Effectively impossible to change via this feature

                current = terminal_values[feature]
                required = current + delta

                counterfactuals.append(
                    Counterfactual(
                        feature=feature,
                        current_value=current,
                        required_value=required,
                        change_magnitude=delta,
                        change_type="flip_decision",
                        description=f"Change {feature} to flip critical decision",
                    )
                )

    else:
        # Fallback: only find counterfactuals for simple cases where
        # the decision directly compares raw terminals
        for d in decisions:
            # Check observation side
            feature = d.observation_source
            if feature in actionable and feature in terminal_values:
                current = terminal_values[feature]
                if abs(current - d.observation_value) < 0.01:  # Direct terminal
                    required = d.tipping_point
                    delta = required - current
                    counterfactuals.append(
                        Counterfactual(
                            feature=feature,
                            current_value=current,
                            required_value=required,
                            change_magnitude=delta,
                            change_type="flip_decision",
                            description=f"Flip decision: {d.observation_source} {d.op.value} {d.threshold_source}",
                        )
                    )

            # Check threshold side
            feature = d.threshold_source
            if feature in actionable and feature in terminal_values:
                current = terminal_values[feature]
                if abs(current - d.threshold_value) < 0.01:  # Direct terminal
                    if d.op in (ComparisonOp.LTE, ComparisonOp.LT):
                        required = (
                            d.observation_value - 1e-9
                            if d.result
                            else d.observation_value + 1e-9
                        )
                    else:
                        required = (
                            d.observation_value + 1e-9
                            if d.result
                            else d.observation_value - 1e-9
                        )
                    delta = required - current
                    counterfactuals.append(
                        Counterfactual(
                            feature=feature,
                            current_value=current,
                            required_value=required,
                            change_magnitude=delta,
                            change_type="flip_decision",
                            description=f"Flip decision by changing: {d.threshold_source}",
                        )
                    )

    # Sort by absolute change magnitude (easiest changes first)
    counterfactuals.sort(key=lambda cf: abs(cf.change_magnitude))

    return counterfactuals


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("FCP Tracker v2 - Self Test")
    print("=" * 60)

    # Create a simple test
    pset = create_tracked_pset("test", ["x", "y"])
    pset.addTerminal(TrackedValue.from_constant(1.0), name="ONE")
    pset.addTerminal(TrackedValue.from_constant(0.0), name="ZERO")

    # Test 1: Simple conditional
    print("\n### Test 1: Simple IFLTE ###")
    tree = gp.PrimitiveTree.from_string("IFLTE(x, y, ONE, ZERO)", pset)
    result = evaluate_with_tracking(tree, pset, {"x": 3.0, "y": 5.0})

    print(f"Tree: {tree}")
    print(f"Inputs: x=3.0, y=5.0")
    print(f"Output: {result.output.value}")
    print(f"Expected: 1.0 (x <= y)")
    assert result.output.value == 1.0, f"Expected 1.0, got {result.output.value}"
    print("✓ Output correct!")

    # Check decision tracking
    assert (
        len(result.decisions) == 1
    ), f"Expected 1 decision, got {len(result.decisions)}"
    d = result.decisions[0]
    print(f"\nDecision: {d.expression}")
    print(f"  Satisfaction: {d.satisfaction:.3f}")
    print(f"  Tipping point: {d.tipping_point:.4f}")
    print(f"  Distance to flip: {d.distance_to_flip:.4f}")
    print("✓ Decision tracking correct!")

    # Test 2: Signed contributions for arithmetic
    print("\n### Test 2: Signed Contributions ###")
    tree2 = gp.PrimitiveTree.from_string("SUB(x, y)", pset)
    result2 = evaluate_with_tracking(tree2, pset, {"x": 10.0, "y": 3.0})

    print(f"Tree: {tree2}")
    print(f"Output: {result2.output.value}")
    print(f"Signed contributions: {result2.output.signed_contributions}")

    signed = result2.output.signed_contributions
    assert (
        abs(signed.total - 7.0) < 0.01
    ), f"Signed should sum to 7.0, got {signed.total}"
    assert signed.contributions.get("x", 0) > 0, "x should have positive contribution"
    assert signed.contributions.get("y", 0) < 0, "y should have negative contribution"
    print("✓ Signed contributions correct!")

    # Test 3: Multiplication
    print("\n### Test 3: Multiplication ###")
    tree3 = gp.PrimitiveTree.from_string("MUL(x, y)", pset)
    result3 = evaluate_with_tracking(tree3, pset, {"x": 2.0, "y": 3.0})

    print(f"Tree: {tree3}")
    print(f"Output: {result3.output.value}")
    print(f"Signed contributions: {result3.output.signed_contributions}")

    signed3 = result3.output.signed_contributions
    assert (
        abs(signed3.total - 6.0) < 0.01
    ), f"Signed should sum to 6.0, got {signed3.total}"
    print("✓ Multiplication tracking correct!")

    # Test 4: Full explanation
    print("\n### Test 4: Full Explanation ###")
    print(result.explain(verbose=True))

    print("\n" + "=" * 60)
    print("✓ All tests passed!")