"""
Forward Contribution Propagation - Jupyter Testing Framework

A modular framework for defining GP problems, test cases, and running
systematic FCP experiments. Designed for interactive Jupyter notebook use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum
import math
from datetime import datetime

# Optional imports for enhanced Jupyter display
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from IPython.display import display, HTML, Markdown

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


# =============================================================================
# Display Helper Functions (for v2 features)
# =============================================================================


def _satisfaction_color(satisfaction: float) -> str:
    """Return color based on satisfaction score."""
    if satisfaction > 0.3:
        return "#4CAF50"  # Green - comfortably satisfied
    elif satisfaction > 0.1:
        return "#8BC34A"  # Light green - satisfied with margin
    elif satisfaction > 0:
        return "#FFC107"  # Yellow - barely satisfied
    elif satisfaction > -0.1:
        return "#FF9800"  # Orange - barely not satisfied
    elif satisfaction > -0.3:
        return "#f44336"  # Red - not satisfied
    else:
        return "#b71c1c"  # Dark red - far from satisfied


def _contribution_color(value: float) -> str:
    """Return color based on contribution sign."""
    if value > 0:
        return "#4CAF50"  # Green for positive
    elif value < 0:
        return "#f44336"  # Red for negative
    else:
        return "#9e9e9e"  # Gray for zero


def _render_signed_contributions_html(
    raw_result: Any, max_contributors: int = 8
) -> str:
    """Generate HTML for signed contributions (SHAP-like waterfall)."""
    if not hasattr(raw_result, "output") or not hasattr(
        raw_result.output, "signed_contributions"
    ):
        return ""

    signed = raw_result.output.signed_contributions
    contribs = signed.contributions
    baseline = signed.baseline
    total = signed.total

    # Sort by absolute value
    sorted_contribs = sorted(
        [(k, v) for k, v in contribs.items() if not k.startswith("_")],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:max_contributors]

    if not sorted_contribs and abs(baseline) < 0.001:
        return ""

    # Find max absolute value for scaling
    max_abs = max(abs(v) for _, v in sorted_contribs) if sorted_contribs else 1
    max_abs = max(max_abs, abs(baseline), 1)

    rows = ""

    # Baseline row
    if abs(baseline) > 0.001:
        width = int(abs(baseline) / max_abs * 150)
        color = "#22c55e" if baseline > 0 else "#ef4444" if baseline < 0 else "#9ca3af"
        rows += f"""
        <tr style="background: #f9fafb;">
            <td style="padding: 6px 10px; font-style: italic; color: #6b7280;">baseline</td>
            <td style="padding: 6px 10px; text-align: right; color: {color};">{baseline:+.4f}</td>
            <td style="padding: 6px 10px;">
                <div style="display: inline-block; width: {width}px; height: 14px; 
                            background: {color}; opacity: 0.5; border-radius: 2px;"></div>
            </td>
        </tr>
        """

    # Contribution rows
    for name, value in sorted_contribs:
        width = int(abs(value) / max_abs * 150)
        color = "#22c55e" if value > 0 else "#ef4444" if value < 0 else "#9ca3af"
        arrow = "↑" if value > 0 else "↓" if value < 0 else "–"
        rows += f"""
        <tr>
            <td style="padding: 6px 10px; color: #1f2937;"><b>{name}</b></td>
            <td style="padding: 6px 10px; text-align: right; color: {color}; font-weight: 500;">{value:+.4f} {arrow}</td>
            <td style="padding: 6px 10px;">
                <div style="display: inline-block; width: {width}px; height: 14px; 
                            background: {color}; border-radius: 2px;"></div>
            </td>
        </tr>
        """

    # Total row
    rows += f"""
    <tr style="border-top: 2px solid #374151;">
        <td style="padding: 6px 10px; color: #1f2937; font-weight: bold;">Total</td>
        <td style="padding: 6px 10px; text-align: right; color: #1f2937; font-weight: bold;">{total:.4f}</td>
        <td style="padding: 6px 10px; color: #6b7280;">= Output</td>
    </tr>
    """

    return f"""
    <div style="margin: 10px 0;">
        <b style="color: #1f2937;">Signed Contributions</b> 
        <span style="color: #6b7280; font-size: 12px;">(sum to output)</span>
        <table style="border-collapse: collapse; margin-top: 8px; font-size: 13px; background: #ffffff;">
            {rows}
        </table>
    </div>
    """


def _render_decision_chain_html(raw_result: Any) -> str:
    """Generate HTML for decision chain visualization."""
    if not hasattr(raw_result, "decisions") or not raw_result.decisions:
        return ""

    # Filter to only active decisions and reverse to show in logical execution order (outer first)
    if hasattr(raw_result, "active_decisions"):
        decisions = list(reversed(raw_result.active_decisions))
    else:
        # Fallback for older versions
        decisions = list(
            reversed([d for d in raw_result.decisions if getattr(d, "active", True)])
        )

    if not decisions:
        return ""

    critical = (
        raw_result.get_critical_decision()
        if hasattr(raw_result, "get_critical_decision")
        else None
    )

    cards = ""
    for i, d in enumerate(decisions):
        is_critical = d is critical

        # Colors - use explicit colors that work on both light and dark backgrounds
        result_color = "#22c55e" if d.result else "#ef4444"  # green-500 / red-500
        border_color = (
            "#f97316" if is_critical else ("#22c55e" if d.result else "#ef4444")
        )
        border_width = "3px" if is_critical else "2px"
        bg_color = "#ffffff"
        text_color = "#1f2937"
        muted_color = "#6b7280"

        # Critical badge
        critical_badge = (
            """<span style="background: #f97316; color: white; padding: 2px 6px; 
                                         border-radius: 3px; font-size: 10px; margin-left: 8px;
                                         font-weight: bold;">CRITICAL</span>"""
            if is_critical
            else ""
        )

        # Branch expressions
        then_expr = getattr(d, "then_expr", "") or "?"
        else_expr = getattr(d, "else_expr", "") or "?"

        # Truncate long expressions
        max_expr_len = 40
        if len(then_expr) > max_expr_len:
            then_expr = then_expr[:max_expr_len] + "..."
        if len(else_expr) > max_expr_len:
            else_expr = else_expr[:max_expr_len] + "..."

        # Build the 2-segment satisfaction bar
        # The bar shows: [RED region] | tipping point | [GREEN region]
        # For LTE: obs <= threshold means GREEN on right (satisfied when obs is low)
        # Marker shows where observation currently is

        obs = d.observation_value
        tipping = d.tipping_point

        # Determine the range for visualization
        # Use a range that shows both the observation and tipping point
        min_val = min(obs, tipping) - abs(tipping - obs) * 0.5
        max_val = max(obs, tipping) + abs(tipping - obs) * 0.5
        if min_val == max_val:
            min_val -= 1
            max_val += 1
        val_range = max_val - min_val

        # Position of tipping point (as percentage)
        tipping_pct = ((tipping - min_val) / val_range) * 100
        tipping_pct = max(5, min(95, tipping_pct))

        # Position of observation (as percentage)
        obs_pct = ((obs - min_val) / val_range) * 100
        obs_pct = max(2, min(98, obs_pct))

        # For LTE: left of tipping is FALSE (red), right is TRUE (green)
        # But we want satisfied (green) to be visually clear
        # If result is TRUE, observation is on the "good" side

        cards += f"""
        <div style="border-left: {border_width} solid {border_color}; 
                    padding: 12px 14px; margin: 8px 0; background: {bg_color};
                    border-radius: 0 6px 6px 0; color: {text_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span style="font-weight: bold;">Decision {i+1}</span>
                    {critical_badge}
                </div>
                <span style="color: {result_color}; font-weight: bold; font-size: 14px;">
                    {'TRUE ✓' if d.result else 'FALSE ✗'}
                </span>
            </div>
            
            <div style="font-family: monospace; font-size: 13px; margin: 8px 0; padding: 6px 8px;
                        background: #f3f4f6; border-radius: 4px; color: #374151;">
                {d.observation_source} <b>{d.op.value}</b> {d.threshold_source}
            </div>
            
            <div style="display: flex; gap: 20px; margin: 10px 0; font-size: 13px;">
                <div style="flex: 1; padding: 6px 10px; background: #dcfce7; border-radius: 4px; border-left: 3px solid #22c55e;">
                    <span style="color: #166534; font-weight: 500;">TRUE →</span>
                    <span style="color: #15803d; font-family: monospace; font-size: 12px;"> {then_expr}</span>
                </div>
                <div style="flex: 1; padding: 6px 10px; background: #fee2e2; border-radius: 4px; border-left: 3px solid #ef4444;">
                    <span style="color: #991b1b; font-weight: 500;">FALSE →</span>
                    <span style="color: #b91c1c; font-family: monospace; font-size: 12px;"> {else_expr}</span>
                </div>
            </div>
            
            <div style="margin-top: 12px;">
                <div style="position: relative; height: 50px;">
                    <!-- The bar: red left, green right, split at tipping point -->
                    <div style="position: absolute; top: 20px; left: 0; right: 0; height: 10px; 
                                display: flex; border-radius: 5px; overflow: hidden;">
                        <div style="width: {tipping_pct}%; background: #fca5a5;"></div>
                        <div style="width: {100 - tipping_pct}%; background: #86efac;"></div>
                    </div>
                    
                    <!-- Tipping point marker (line) -->
                    <div style="position: absolute; top: 15px; left: {tipping_pct}%; transform: translateX(-50%);
                                width: 2px; height: 20px; background: #374151;"></div>
                    
                    <!-- Tipping point label -->
                    <div style="position: absolute; top: 36px; left: {tipping_pct}%; transform: translateX(-50%);
                                font-size: 10px; color: {muted_color}; white-space: nowrap;">
                        tipping: {tipping:.4g}
                    </div>
                    
                    <!-- Observation marker (circle) -->
                    <div style="position: absolute; top: 17px; left: {obs_pct}%; transform: translateX(-50%);
                                width: 16px; height: 16px; background: {result_color}; 
                                border: 2px solid white; border-radius: 50%; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>
                    
                    <!-- Observation label -->
                    <div style="position: absolute; top: 0px; left: {obs_pct}%; transform: translateX(-50%);
                                font-size: 11px; color: {text_color}; font-weight: 500; white-space: nowrap;">
                        obs: {obs:.4g}
                    </div>
                </div>
            </div>
            
            <div style="display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: {muted_color};">
                <div>
                    <span>Flip dist:</span>
                    <b style="color: {text_color};">{d.distance_to_flip:+.4g}</b>
                </div>
                <div>
                    <span>Power:</span>
                    <b style="color: {text_color};">{d.power:.0%}</b>
                </div>
            </div>
        </div>
        """

    return f"""
    <div style="margin: 10px 0;">
        <b style="color: #1f2937;">Decision Chain</b> 
        <span style="color: #6b7280; font-size: 12px;">({len(decisions)} decision{'s' if len(decisions) != 1 else ''})</span>
        {cards}
    </div>
    """


# =============================================================================
# Core Data Classes
# =============================================================================


@dataclass
class TestCase:
    """A single test case with input values and optional expected outcomes."""

    name: str
    inputs: Dict[str, float]
    expected_output: Optional[float] = None
    expected_class: Optional[int] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            raise ValueError("TestCase must have a name")
        if not self.inputs:
            raise ValueError("TestCase must have inputs")

    def _repr_html_(self) -> str:
        """Rich display for Jupyter."""
        rows = [
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in self.inputs.items()
        ]
        html = f"""
        <div style="margin: 10px 0;">
            <b style="font-size: 14px;">{self.name}</b>
            {f'<br><i>{self.description}</i>' if self.description else ''}
            <table style="margin-top: 5px; border-collapse: collapse;">
                {''.join(rows)}
            </table>
            {f'<br>Expected: {self.expected_class if self.expected_class is not None else self.expected_output}' if self.expected_output is not None or self.expected_class is not None else ''}
            {f'<br>Tags: {", ".join(self.tags)}' if self.tags else ''}
        </div>
        """
        return html

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "name": self.name,
            **self.inputs,
            "expected_output": self.expected_output,
            "expected_class": self.expected_class,
            "tags": ", ".join(self.tags),
        }


@dataclass
class AttributionResult:
    """Results from running FCP attribution on a single test case."""

    test_case: TestCase
    output_value: float
    value_contributions: Dict[str, float]
    decision_contributions: Dict[str, float]
    decision_margins: List[Dict[str, Any]] = field(default_factory=list)
    raw_result: Any = None  # Store the original TrackedValue result
    tree_expression: Optional[str] = None  # Store the tree expression for display

    @property
    def top_value_contributors(self) -> List[Tuple[str, float]]:
        """Return value contributions sorted by magnitude."""
        return sorted(
            [
                (k, v)
                for k, v in self.value_contributions.items()
                if not k.startswith("_")
            ],
            key=lambda x: -x[1],
        )

    @property
    def top_decision_contributors(self) -> List[Tuple[str, float]]:
        """Return decision contributions sorted by magnitude."""
        return sorted(
            [
                (k, v)
                for k, v in self.decision_contributions.items()
                if not k.startswith("_")
            ],
            key=lambda x: -x[1],
        )

    @property
    def predicted_class(self) -> Optional[int]:
        """For binary classification, return 0 or 1."""
        if self.output_value in (0.0, 1.0):
            return int(self.output_value)
        return None

    @property
    def min_decision_margin(self) -> Optional[float]:
        """Return the smallest absolute margin (closest decision point)."""
        if not self.decision_margins:
            return None
        return min(abs(m.get("margin", float("inf"))) for m in self.decision_margins)

    @property
    def is_correct(self) -> Optional[bool]:
        """Check if prediction matches expected (for classification or regression)."""
        if self.test_case.expected_class is not None:
            return self.predicted_class == self.test_case.expected_class
        elif self.test_case.expected_output is not None:
            return abs(self.output_value - self.test_case.expected_output) < 1e-6
        return None

    def _repr_html_(self) -> str:
        """Rich display for Jupyter with v2 features."""
        # Tree expression section
        tree_html = ""
        if self.tree_expression:
            tree_html = f"""
            <div style="background: #f3f4f6; padding: 8px 12px; margin-bottom: 10px; 
                        border-radius: 4px; font-family: monospace; font-size: 12px;
                        overflow-x: auto; white-space: nowrap; color: #374151;">
                <b>Tree:</b> {self.tree_expression}
            </div>
            """

        # Input values section
        inputs_html = " | ".join(
            [
                f"<b>{k}</b>={v:.4g}" if isinstance(v, float) else f"<b>{k}</b>={v}"
                for k, v in self.test_case.inputs.items()
            ]
        )

        # Expected vs actual
        expected = (
            self.test_case.expected_class
            if self.test_case.expected_class is not None
            else self.test_case.expected_output
        )
        if expected is not None:
            correct = self.is_correct
            if correct is True:
                status = "✓"
                status_color = "#22c55e"
            elif correct is False:
                status = "✗"
                status_color = "#ef4444"
            else:
                status = "?"
                status_color = "#9ca3af"
            expected_html = f"""
            <span style="margin-left: 15px;">
                <b>Expected:</b> {expected} 
                <span style="color: {status_color}; font-weight: bold;">{status}</span>
            </span>
            """
        else:
            expected_html = ""

        # Description if present
        desc_html = ""
        if self.test_case.description:
            desc_html = f'<div style="color: #6b7280; font-style: italic; margin-bottom: 8px;">{self.test_case.description}</div>'

        # Signed contributions (v2 feature)
        signed_html = ""
        if self.raw_result:
            signed_html = _render_signed_contributions_html(self.raw_result)

        # Decision chain (v2 feature)
        decision_chain_html = ""
        if self.raw_result:
            decision_chain_html = _render_decision_chain_html(self.raw_result)

        # Normalized contributions (legacy display, shown if no v2 features available)
        normalized_html = ""
        if not signed_html and not decision_chain_html:
            # Fall back to normalized contributions display
            value_bars = ""
            for name, contrib in self.top_value_contributors[:5]:
                width = int(contrib * 200)
                value_bars += f"""
                <div style="margin: 2px 0;">
                    <span style="display: inline-block; width: 70px; color: #1f2937;">{name}</span>
                    <span style="display: inline-block; width: {width}px; height: 16px; 
                          background: #22c55e; margin-right: 5px; border-radius: 2px;"></span>
                    <span style="color: #374151;">{contrib:.1%}</span>
                </div>
                """

            decision_bars = ""
            if self.decision_contributions:
                for name, contrib in self.top_decision_contributors[:5]:
                    width = int(contrib * 200)
                    decision_bars += f"""
                    <div style="margin: 2px 0;">
                        <span style="display: inline-block; width: 70px; color: #1f2937;">{name}</span>
                        <span style="display: inline-block; width: {width}px; height: 16px; 
                              background: #3b82f6; margin-right: 5px; border-radius: 2px;"></span>
                        <span style="color: #374151;">{contrib:.1%}</span>
                    </div>
                    """

            normalized_html = f"""
            <div style="display: flex; gap: 40px;">
                <div>
                    <b style="color: #1f2937;">Value Contributions</b>
                    {value_bars if value_bars else '<i style="color: #9ca3af;">None</i>'}
                </div>
                <div>
                    <b style="color: #1f2937;">Decision Contributions</b>
                    {decision_bars if decision_bars else '<i style="color: #9ca3af;">None</i>'}
                </div>
            </div>
            """

        html = f"""
        <div style="border: 1px solid #d1d5db; padding: 14px; margin: 10px 0; border-radius: 6px; 
                    background: #ffffff; color: #1f2937;">
            <h4 style="margin: 0 0 8px 0; color: #111827;">{self.test_case.name}</h4>
            {desc_html}
            {tree_html}
            <div style="background: #f9fafb; padding: 8px 12px; margin-bottom: 10px; 
                        border-radius: 4px; font-size: 13px; color: #374151;">
                {inputs_html}
            </div>
            <div style="margin-bottom: 10px;">
                <b>Output:</b> <span style="font-size: 16px; color: #111827;">{self.output_value:.6g}</span>{expected_html}
            </div>
            {signed_html}
            {decision_chain_html}
            {normalized_html}
        </div>
        """
        return html

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "name": self.test_case.name,
            "output": self.output_value,
            "expected": self.test_case.expected_output or self.test_case.expected_class,
            "correct": self.is_correct,
            "min_margin": self.min_decision_margin,
        }
        # Add input values
        for k, v in self.test_case.inputs.items():
            result[f"in_{k}"] = v
        # Add contributions with prefixes
        for k, v in self.value_contributions.items():
            if not k.startswith("_"):
                result[f"val_{k}"] = v
        for k, v in self.decision_contributions.items():
            if not k.startswith("_"):
                result[f"dec_{k}"] = v
        return result


@dataclass
class CounterfactualResult:
    """Results from counterfactual search."""

    original_case: TestCase
    success: bool
    modified_inputs: Optional[Dict[str, float]] = None
    new_output: Optional[float] = None
    perturbation_magnitude: Optional[float] = None
    iterations_used: int = 0
    terminals_modified: List[str] = field(default_factory=list)

    @property
    def changes(self) -> Dict[str, float]:
        """Return dict of terminal -> delta for modified values."""
        if not self.success or not self.modified_inputs:
            return {}
        return {
            k: self.modified_inputs[k] - self.original_case.inputs[k]
            for k in self.terminals_modified
            if k in self.modified_inputs and k in self.original_case.inputs
        }

    def _repr_html_(self) -> str:
        """Rich display for Jupyter."""
        if self.success:
            changes_html = ""
            for terminal, delta in self.changes.items():
                orig = self.original_case.inputs[terminal]
                new = self.modified_inputs[terminal]
                color = "green" if delta > 0 else "red"
                changes_html += f"""
                <tr>
                    <td>{terminal}</td>
                    <td>{orig:.4f}</td>
                    <td>{new:.4f}</td>
                    <td style="color: {color}">{delta:+.4f}</td>
                </tr>
                """

            html = f"""
            <div style="border: 1px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h4 style="color: #4CAF50; margin: 0;">✓ Counterfactual Found</h4>
                <p><b>Original output:</b> {self.original_case.expected_class or self.original_case.expected_output} 
                   → <b>New output:</b> {self.new_output}</p>
                <table style="border-collapse: collapse;">
                    <tr><th>Terminal</th><th>Original</th><th>New</th><th>Δ</th></tr>
                    {changes_html}
                </table>
                <p><b>Perturbation (L2):</b> {self.perturbation_magnitude:.4f}</p>
            </div>
            """
        else:
            html = f"""
            <div style="border: 1px solid #f44336; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h4 style="color: #f44336; margin: 0;">✗ Counterfactual Not Found</h4>
                <p>Could not find valid counterfactual after {self.iterations_used} iterations.</p>
            </div>
            """
        return html


class ProblemType(Enum):
    """Classification of GP problem types."""

    SYMBOLIC_REGRESSION = "symbolic_regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


# =============================================================================
# Problem Definition Base Class
# =============================================================================


class GPProblem(ABC):
    """
    Abstract base class for defining a GP problem with its primitive set,
    ideal solution (if known), and test cases.
    """

    def __init__(self, name: str, problem_type: ProblemType):
        self.name = name
        self.problem_type = problem_type
        self._test_cases: List[TestCase] = []
        self._pset = None
        self._tree = None

    @property
    @abstractmethod
    def input_terminals(self) -> List[str]:
        """Return list of input terminal names."""
        pass

    @property
    @abstractmethod
    def constants(self) -> Dict[str, float]:
        """Return dict of constant name -> value."""
        pass

    @property
    def include_conditionals(self) -> bool:
        """Whether to include IFLTE and comparison operators."""
        return True

    @property
    @abstractmethod
    def ideal_expression(self) -> Optional[str]:
        """Return the ideal tree expression string, or None if unknown."""
        pass

    @abstractmethod
    def ground_truth(self, inputs: Dict[str, float]) -> float:
        """Compute the ground-truth output for given inputs."""
        pass

    def add_test_case(self, case: TestCase) -> "GPProblem":
        """Add a test case. Returns self for chaining."""
        missing = set(self.input_terminals) - set(case.inputs.keys())
        if missing:
            raise ValueError(f"Test case '{case.name}' missing inputs: {missing}")
        self._test_cases.append(case)
        return self

    def add_test_cases(self, cases: List[TestCase]) -> "GPProblem":
        """Add multiple test cases. Returns self for chaining."""
        for case in cases:
            self.add_test_case(case)
        return self

    @property
    def test_cases(self) -> List[TestCase]:
        return self._test_cases.copy()

    def get_cases_by_tag(self, tag: str) -> List[TestCase]:
        """Return test cases matching a specific tag."""
        return [tc for tc in self._test_cases if tag in tc.tags]

    def cases_df(self) -> "pd.DataFrame":
        """Return test cases as a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for DataFrame output")
        return pd.DataFrame([tc.to_dict() for tc in self._test_cases])

    def setup(
        self, create_pset_fn: Callable, tracked_value_class: Any, gp_module: Any
    ) -> Tuple[Any, Any]:
        """
        One-step setup: create primitive set and build ideal tree.

        Args:
            create_pset_fn: Your create_tracked_pset function
            tracked_value_class: Your TrackedValue class
            gp_module: The DEAP gp module

        Returns:
            (pset, tree) tuple
        """
        # Create primitive set
        self._pset = create_pset_fn(
            self.name,
            self.input_terminals,
            include_conditionals=self.include_conditionals,
        )

        # Add constants as TrackedValues
        for const_name, const_value in self.constants.items():
            self._pset.addTerminal(
                tracked_value_class.from_constant(const_value, name=const_name),
                name=const_name,
            )

        # Build ideal tree
        if self.ideal_expression:
            self._tree = gp_module.PrimitiveTree.from_string(
                self.ideal_expression, self._pset
            )

        return self._pset, self._tree

    def _repr_html_(self) -> str:
        """Rich display for Jupyter."""
        html = f"""
        <div style="border: 2px solid #333; padding: 15px; margin: 10px 0; border-radius: 8px;">
            <h3 style="margin: 0 0 10px 0;">{self.name}</h3>
            <p><b>Type:</b> {self.problem_type.value}</p>
            <p><b>Inputs:</b> {', '.join(self.input_terminals)}</p>
            <p><b>Constants:</b> {self.constants}</p>
            <p><b>Test cases:</b> {len(self._test_cases)}</p>
            <details>
                <summary>Ideal Expression</summary>
                <code style="display: block; padding: 10px; background: #f5f5f5; margin-top: 5px;">
                    {self.ideal_expression}
                </code>
            </details>
        </div>
        """
        return html

    def __repr__(self) -> str:
        return f"GPProblem({self.name}, {len(self._test_cases)} test cases)"


# =============================================================================
# Test Suite Results
# =============================================================================


@dataclass
class TestSuiteResults:
    """Aggregated results from running a test suite."""

    problem_name: str
    timestamp: str
    results: List[AttributionResult]
    counterfactual_results: List[CounterfactualResult] = field(default_factory=list)
    tree_expression: Optional[str] = None

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> Optional[float]:
        correct = sum(
            1
            for r in self.results
            if r.test_case.expected_class is not None
            and r.predicted_class == r.test_case.expected_class
        )
        total = sum(1 for r in self.results if r.test_case.expected_class is not None)
        return correct / total if total > 0 else None

    @property
    def mean_absolute_error(self) -> Optional[float]:
        errors = [
            abs(r.output_value - r.test_case.expected_output)
            for r in self.results
            if r.test_case.expected_output is not None
        ]
        return sum(errors) / len(errors) if errors else None

    @property
    def counterfactual_success_rate(self) -> Optional[float]:
        if not self.counterfactual_results:
            return None
        successes = sum(1 for cf in self.counterfactual_results if cf.success)
        return successes / len(self.counterfactual_results)

    @property
    def num_correct(self) -> int:
        return sum(1 for r in self.results if r.is_correct is True)

    @property
    def num_incorrect(self) -> int:
        return sum(1 for r in self.results if r.is_correct is False)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert all results to a DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for DataFrame output")
        return pd.DataFrame([r.to_dict() for r in self.results])

    def _repr_html_(self) -> str:
        """Rich display for Jupyter."""
        metrics = []
        if self.accuracy is not None:
            metrics.append(
                f"<b>Accuracy:</b> {self.accuracy:.1%} ({self.num_correct}/{self.total_cases})"
            )
        if self.mean_absolute_error is not None:
            metrics.append(f"<b>MAE:</b> {self.mean_absolute_error:.4f}")
        if self.counterfactual_success_rate is not None:
            metrics.append(f"<b>CF Success:</b> {self.counterfactual_success_rate:.1%}")

        tree_html = ""
        if self.tree_expression:
            tree_html = f"""
            <div style="background: #f8f8f8; padding: 8px; margin: 10px 0; 
                        border-radius: 4px; font-family: monospace; font-size: 12px;
                        overflow-x: auto; white-space: nowrap;">
                <b>Tree:</b> {self.tree_expression}
            </div>
            """

        html = f"""
        <div style="border: 2px solid #2196F3; padding: 15px; margin: 10px 0; border-radius: 8px;">
            <h3 style="margin: 0 0 10px 0;">Results: {self.problem_name}</h3>
            {tree_html}
            <p><b>Cases:</b> {self.total_cases} | {' | '.join(metrics)}</p>
            <p style="color: #666; font-size: 12px;">{self.timestamp}</p>
        </div>
        """
        return html


# =============================================================================
# Test Runner
# =============================================================================


class TestRunner:
    """
    Runs test cases through the FCP attribution system.

    Usage:
        runner = TestRunner(evaluate_with_tracking, find_counterfactual)
        results = runner.run(problem, tree, pset)
    """

    def __init__(
        self, evaluate_fn: Callable, counterfactual_fn: Optional[Callable] = None
    ):
        """
        Args:
            evaluate_fn: Function like evaluate_with_tracking(tree, pset, inputs)
            counterfactual_fn: Optional function like find_counterfactual(...)
        """
        self.evaluate_fn = evaluate_fn
        self.counterfactual_fn = counterfactual_fn

    def run_single(
        self, tree: Any, pset: Any, test_case: TestCase, include_tree_expr: bool = True
    ) -> AttributionResult:
        """Run a single test case and return attribution results."""
        result = self.evaluate_fn(tree, pset, test_case.inputs)

        # Get tree expression string
        tree_expr = str(tree) if include_tree_expr else None

        return AttributionResult(
            test_case=test_case,
            output_value=result.output.value,
            value_contributions=dict(result.output.contributions),
            decision_contributions=dict(result.output.decision_contributions),
            decision_margins=getattr(result, "decision_margins", []),
            raw_result=result,
            tree_expression=tree_expr,
        )

    def run(
        self,
        problem: GPProblem,
        tree: Any = None,
        pset: Any = None,
        cases: Optional[List[TestCase]] = None,
        tags: Optional[List[str]] = None,
        run_counterfactuals: bool = False,
        counterfactual_terminals: Optional[List[str]] = None,
        max_cf_iterations: int = 50,
        include_tree_expr: bool = True,
    ) -> TestSuiteResults:
        """
        Run test cases and return results.

        Args:
            problem: The GPProblem instance
            tree: GP tree (uses problem._tree if not provided)
            pset: Primitive set (uses problem._pset if not provided)
            cases: Specific cases to run (default: all)
            tags: Filter cases by tags
            run_counterfactuals: Whether to search for counterfactuals
            counterfactual_terminals: Which terminals can be modified
            max_cf_iterations: Max iterations for counterfactual search
            include_tree_expr: Whether to include tree expression in results
        """
        tree = tree or problem._tree
        pset = pset or problem._pset

        if tree is None or pset is None:
            raise ValueError(
                "Must provide tree and pset, or call problem.setup() first"
            )

        # Get tree expression string
        tree_expr = str(tree) if include_tree_expr else None

        # Select test cases
        if cases is not None:
            test_cases = cases
        elif tags is not None:
            test_cases = [
                tc for tc in problem.test_cases if any(tag in tc.tags for tag in tags)
            ]
        else:
            test_cases = problem.test_cases

        results = []
        cf_results = []

        for case in test_cases:
            # Run attribution
            attr_result = self.run_single(
                tree, pset, case, include_tree_expr=include_tree_expr
            )
            results.append(attr_result)

            # Run counterfactual search if requested
            if run_counterfactuals and self.counterfactual_fn:
                terminals = counterfactual_terminals or problem.input_terminals
                cf_result = self._find_counterfactual(
                    tree, pset, case, terminals, max_cf_iterations
                )
                cf_results.append(cf_result)

        return TestSuiteResults(
            problem_name=problem.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=results,
            counterfactual_results=cf_results,
            tree_expression=tree_expr,
        )

    def _find_counterfactual(
        self,
        tree: Any,
        pset: Any,
        test_case: TestCase,
        terminals_to_modify: List[str],
        max_iterations: int,
    ) -> CounterfactualResult:
        """Find a counterfactual for a given test case."""
        cf_values, cf_result, success = self.counterfactual_fn(
            tree,
            pset,
            test_case.inputs,
            terminals_to_modify=terminals_to_modify,
            max_iterations=max_iterations,
        )

        if success:
            perturbation = math.sqrt(
                sum(
                    (cf_values[t] - test_case.inputs[t]) ** 2
                    for t in terminals_to_modify
                    if t in cf_values and t in test_case.inputs
                )
            )

            return CounterfactualResult(
                original_case=test_case,
                success=True,
                modified_inputs=cf_values,
                new_output=cf_result.output.value,
                perturbation_magnitude=perturbation,
                iterations_used=max_iterations,
                terminals_modified=terminals_to_modify,
            )
        else:
            return CounterfactualResult(
                original_case=test_case,
                success=False,
                iterations_used=max_iterations,
                terminals_modified=terminals_to_modify,
            )


# =============================================================================
# Built-in Problems
# =============================================================================


class PointInCircleProblem(GPProblem):
    """
    Binary classification: Is point (px, py) inside circle at (cx, cy) with radius r?
    """

    def __init__(self, include_default_cases: bool = True):
        super().__init__("point_in_circle", ProblemType.BINARY_CLASSIFICATION)
        if include_default_cases:
            self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["cx", "cy", "r", "px", "py"]

    @property
    def constants(self) -> Dict[str, float]:
        return {"ZERO": 0.0, "ONE": 1.0}

    @property
    def ideal_expression(self) -> str:
        return (
            "IFLTE(SQRT(ADD(SQUARE(SUB(cx, px)), SQUARE(SUB(cy, py)))), r, ONE, ZERO)"
        )

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        cx, cy, r = inputs["cx"], inputs["cy"], inputs["r"]
        px, py = inputs["px"], inputs["py"]
        distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        return 1.0 if distance <= r else 0.0

    def distance_to_boundary(self, inputs: Dict[str, float]) -> float:
        """Signed distance to boundary (negative = inside)."""
        cx, cy, r = inputs["cx"], inputs["cy"], inputs["r"]
        px, py = inputs["px"], inputs["py"]
        return math.sqrt((cx - px) ** 2 + (cy - py) ** 2) - r

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "inside_comfortable",
                {"cx": 5.0, "cy": 5.0, "r": 3.0, "px": 6.0, "py": 5.5},
                expected_class=1,
                tags=["inside", "standard"],
            ),
            TestCase(
                "outside_comfortable",
                {"cx": 5.0, "cy": 5.0, "r": 3.0, "px": 10.0, "py": 5.0},
                expected_class=0,
                tags=["outside", "standard"],
            ),
            TestCase(
                "outside_barely",
                {"cx": 5.0, "cy": 5.0, "r": 3.0, "px": 8.1, "py": 5.0},
                expected_class=0,
                tags=["outside", "boundary"],
            ),
            TestCase(
                "inside_barely",
                {"cx": 5.0, "cy": 5.0, "r": 3.0, "px": 7.9, "py": 5.0},
                expected_class=1,
                tags=["inside", "boundary"],
            ),
            TestCase(
                "on_boundary",
                {"cx": 5.0, "cy": 5.0, "r": 3.0, "px": 8.0, "py": 5.0},
                expected_class=1,
                tags=["boundary", "edge_case"],
            ),
            TestCase(
                "at_center",
                {"cx": 5.0, "cy": 5.0, "r": 3.0, "px": 5.0, "py": 5.0},
                expected_class=1,
                tags=["inside", "edge_case"],
            ),
            TestCase(
                "x_dominant",
                {"cx": 0.0, "cy": 0.0, "r": 5.0, "px": 4.0, "py": 0.5},
                expected_class=1,
                tags=["inside", "attribution_test"],
                description="X-displacement dominates",
            ),
            TestCase(
                "y_dominant",
                {"cx": 0.0, "cy": 0.0, "r": 5.0, "px": 0.5, "py": 4.0},
                expected_class=1,
                tags=["inside", "attribution_test"],
                description="Y-displacement dominates",
            ),
            TestCase(
                "large_radius",
                {"cx": 0.0, "cy": 0.0, "r": 100.0, "px": 10.0, "py": 10.0},
                expected_class=1,
                tags=["inside", "attribution_test"],
                description="Radius dominates decision",
            ),
            TestCase(
                "small_radius",
                {"cx": 0.0, "cy": 0.0, "r": 1.0, "px": 10.0, "py": 10.0},
                expected_class=0,
                tags=["outside", "attribution_test"],
                description="Displacement dominates decision",
            ),
        ]
        self.add_test_cases(cases)


class WeightedSumProblem(GPProblem):
    """
    Symbolic regression: f(x1, x2, x3) = w1*x1 + w2*x2 + w3*x3

    Tests magnitude-weighted additive merging.
    Ground truth: attribution should reflect |weight * value| for each input.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("weighted_sum", ProblemType.SYMBOLIC_REGRESSION)
        self.weights = weights or {"x1": 3.0, "x2": 2.0, "x3": 1.0}
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return list(self.weights.keys())

    @property
    def constants(self) -> Dict[str, float]:
        return {f"w{i+1}": w for i, w in enumerate(self.weights.values())}

    @property
    def ideal_expression(self) -> str:
        terms = [f"MUL(w{i+1}, {name})" for i, name in enumerate(self.weights.keys())]
        expr = terms[-1]
        for term in reversed(terms[:-1]):
            expr = f"ADD({term}, {expr})"
        return expr

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return sum(self.weights[k] * inputs[k] for k in self.weights)

    def expected_attribution(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Compute expected attribution based on |weight * value|.
        Useful for validating FCP output.
        """
        magnitudes = {k: abs(self.weights[k] * inputs[k]) for k in self.weights}
        total = sum(magnitudes.values())
        if total == 0:
            return {k: 0.0 for k in self.weights}
        return {k: v / total for k, v in magnitudes.items()}

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "equal_inputs",
                {"x1": 1.0, "x2": 1.0, "x3": 1.0},
                expected_output=6.0,
                tags=["attribution_test"],
                description="Attribution should reflect weights: x1=50%, x2=33%, x3=17%",
            ),
            TestCase(
                "x1_dominant",
                {"x1": 10.0, "x2": 1.0, "x3": 1.0},
                expected_output=33.0,
                tags=["attribution_test"],
                description="x1 should dominate attribution (~91%)",
            ),
            TestCase(
                "x3_high_value",
                {"x1": 1.0, "x2": 1.0, "x3": 10.0},
                expected_output=15.0,
                tags=["attribution_test"],
                description="Low-weight x3 compensates with high value (~67%)",
            ),
            TestCase(
                "zero_input",
                {"x1": 0.0, "x2": 5.0, "x3": 5.0},
                expected_output=15.0,
                tags=["edge_case"],
                description="x1 should get 0% attribution",
            ),
            TestCase(
                "mixed_signs",
                {"x1": 2.0, "x2": -3.0, "x3": 1.0},
                expected_output=1.0,
                tags=["attribution_test"],
                description="Negative values still contribute by magnitude",
            ),
            TestCase(
                "all_equal_contribution",
                {"x1": 1.0, "x2": 1.5, "x3": 3.0},
                expected_output=9.0,
                tags=["attribution_test"],
                description="Designed so all three contribute equally (3*1=2*1.5=1*3)",
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Multiplicative Problems - Test Log-Space Weighting
# =============================================================================


class MultiplicativeProblem(GPProblem):
    """
    Symbolic regression: f(x, y, z) = x * y * z

    Tests log-space weighting for multiplicative interactions.
    In multiplication, a 10x factor should contribute equally regardless of
    which operand it appears in (unlike additive where absolute value matters).
    """

    def __init__(self):
        super().__init__("multiplicative", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["x", "y", "z"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def ideal_expression(self) -> str:
        return "MUL(x, MUL(y, z))"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return inputs["x"] * inputs["y"] * inputs["z"]

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "equal_values",
                {"x": 2.0, "y": 2.0, "z": 2.0},
                expected_output=8.0,
                tags=["attribution_test"],
                description="Equal values → equal attribution (~33% each)",
            ),
            TestCase(
                "x_large",
                {"x": 10.0, "y": 1.0, "z": 1.0},
                expected_output=10.0,
                tags=["attribution_test"],
                description="x=10 vs y,z=1: log(10) vs log(1), x dominates",
            ),
            TestCase(
                "z_large",
                {"x": 1.0, "y": 1.0, "z": 10.0},
                expected_output=10.0,
                tags=["attribution_test"],
                description="Same as x_large but z dominates - should be symmetric",
            ),
            TestCase(
                "scaling_test",
                {"x": 10.0, "y": 10.0, "z": 1.0},
                expected_output=100.0,
                tags=["attribution_test"],
                description="x and y equal, both larger than z",
            ),
            TestCase(
                "large_vs_small",
                {"x": 100.0, "y": 1.0, "z": 1.0},
                expected_output=100.0,
                tags=["attribution_test"],
                description="Tests log-space: 100 vs 1 should show strong x dominance",
            ),
            TestCase(
                "fractional",
                {"x": 0.5, "y": 2.0, "z": 4.0},
                expected_output=4.0,
                tags=["attribution_test"],
                description="Mixed scales with fraction",
            ),
            TestCase(
                "near_zero",
                {"x": 0.001, "y": 1000.0, "z": 1.0},
                expected_output=1.0,
                tags=["edge_case"],
                description="Extreme scale difference",
            ),
        ]
        self.add_test_cases(cases)


class MixedArithmeticProblem(GPProblem):
    """
    Symbolic regression: f(a, b, c, d) = (a + b) * (c + d)

    Tests interaction between additive and multiplicative attribution.
    The sums (a+b) and (c+d) each contribute multiplicatively,
    while within each sum the terms contribute additively.
    """

    def __init__(self):
        super().__init__("mixed_arithmetic", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["a", "b", "c", "d"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def ideal_expression(self) -> str:
        return "MUL(ADD(a, b), ADD(c, d))"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return (inputs["a"] + inputs["b"]) * (inputs["c"] + inputs["d"])

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "all_equal",
                {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0},
                expected_output=4.0,
                tags=["attribution_test"],
                description="Symmetric - all should contribute equally (25%)",
            ),
            TestCase(
                "a_dominant_in_sum",
                {"a": 10.0, "b": 1.0, "c": 1.0, "d": 1.0},
                expected_output=22.0,
                tags=["attribution_test"],
                description="a dominates (a+b) additively, but (a+b) also larger multiplicatively",
            ),
            TestCase(
                "left_sum_larger",
                {"a": 5.0, "b": 5.0, "c": 1.0, "d": 1.0},
                expected_output=20.0,
                tags=["attribution_test"],
                description="Left sum (10) vs right sum (2) - affects multiplicative weighting",
            ),
            TestCase(
                "balanced_sums_unequal_terms",
                {"a": 8.0, "b": 2.0, "c": 2.0, "d": 8.0},
                expected_output=100.0,
                tags=["attribution_test"],
                description="Both sums equal (10), but internal distribution differs",
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Conditional Problems - Test Dual-Channel Attribution
# =============================================================================


class SimpleConditionalProblem(GPProblem):
    """
    Binary classification: f(x, threshold, a, b) = a if x <= threshold else b

    Tests basic dual-channel attribution:
    - Decision contribution should be dominated by x and threshold
    - Value contribution should come entirely from a or b (whichever branch taken)
    """

    def __init__(self):
        super().__init__("simple_conditional", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["x", "threshold", "a", "b"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def ideal_expression(self) -> str:
        return "IFLTE(x, threshold, a, b)"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        if inputs["x"] <= inputs["threshold"]:
            return inputs["a"]
        return inputs["b"]

    def decision_margin(self, inputs: Dict[str, float]) -> float:
        """Positive = condition true, negative = condition false."""
        return inputs["threshold"] - inputs["x"]

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "condition_true_comfortable",
                {"x": 3.0, "threshold": 10.0, "a": 100.0, "b": 0.0},
                expected_output=100.0,
                tags=["decision_test"],
                description="x < threshold: value from a, decision from x & threshold",
            ),
            TestCase(
                "condition_false_comfortable",
                {"x": 15.0, "threshold": 10.0, "a": 100.0, "b": 0.0},
                expected_output=0.0,
                tags=["decision_test"],
                description="x > threshold: value from b, decision from x & threshold",
            ),
            TestCase(
                "condition_true_barely",
                {"x": 9.9, "threshold": 10.0, "a": 100.0, "b": 0.0},
                expected_output=100.0,
                tags=["decision_test", "boundary"],
                description="Just barely true - small margin",
            ),
            TestCase(
                "condition_false_barely",
                {"x": 10.1, "threshold": 10.0, "a": 100.0, "b": 0.0},
                expected_output=0.0,
                tags=["decision_test", "boundary"],
                description="Just barely false - small margin",
            ),
            TestCase(
                "large_x_dominates_decision",
                {"x": 100.0, "threshold": 1.0, "a": 5.0, "b": 10.0},
                expected_output=10.0,
                tags=["decision_test", "attribution_test"],
                description="x much larger than threshold in decision contribution",
            ),
            TestCase(
                "large_threshold_dominates_decision",
                {"x": 1.0, "threshold": 100.0, "a": 5.0, "b": 10.0},
                expected_output=5.0,
                tags=["decision_test", "attribution_test"],
                description="threshold much larger than x in decision contribution",
            ),
            TestCase(
                "equal_decision_contributors",
                {"x": 5.0, "threshold": 5.0, "a": 100.0, "b": 0.0},
                expected_output=100.0,
                tags=["decision_test", "boundary", "edge_case"],
                description="x equals threshold - exactly on boundary",
            ),
        ]
        self.add_test_cases(cases)


class NestedConditionalProblem(GPProblem):
    """
    Nested conditionals:
    f(x, y, t1, t2, a, b, c) =
        if x <= t1:
            if y <= t2: a else b
        else:
            c

    Tests decision attribution through nested branching.
    The outer decision depends on x & t1, inner depends on y & t2.
    """

    def __init__(self):
        super().__init__("nested_conditional", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["x", "y", "t1", "t2", "a", "b", "c"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def ideal_expression(self) -> str:
        # IFLTE(x, t1, IFLTE(y, t2, a, b), c)
        return "IFLTE(x, t1, IFLTE(y, t2, a, b), c)"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        if inputs["x"] <= inputs["t1"]:
            if inputs["y"] <= inputs["t2"]:
                return inputs["a"]
            return inputs["b"]
        return inputs["c"]

    def which_branch(self, inputs: Dict[str, float]) -> str:
        """Return which leaf was selected."""
        if inputs["x"] <= inputs["t1"]:
            if inputs["y"] <= inputs["t2"]:
                return "a"
            return "b"
        return "c"

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "outer_true_inner_true",
                {
                    "x": 1.0,
                    "t1": 5.0,
                    "y": 1.0,
                    "t2": 5.0,
                    "a": 100.0,
                    "b": 50.0,
                    "c": 0.0,
                },
                expected_output=100.0,
                tags=["decision_test"],
                description="Both conditions true → a. Decision involves x, t1, y, t2",
            ),
            TestCase(
                "outer_true_inner_false",
                {
                    "x": 1.0,
                    "t1": 5.0,
                    "y": 10.0,
                    "t2": 5.0,
                    "a": 100.0,
                    "b": 50.0,
                    "c": 0.0,
                },
                expected_output=50.0,
                tags=["decision_test"],
                description="Outer true, inner false → b",
            ),
            TestCase(
                "outer_false",
                {
                    "x": 10.0,
                    "t1": 5.0,
                    "y": 1.0,
                    "t2": 5.0,
                    "a": 100.0,
                    "b": 50.0,
                    "c": 0.0,
                },
                expected_output=0.0,
                tags=["decision_test"],
                description="Outer false → c. Inner condition not evaluated",
            ),
            TestCase(
                "outer_barely_true",
                {
                    "x": 4.9,
                    "t1": 5.0,
                    "y": 1.0,
                    "t2": 5.0,
                    "a": 100.0,
                    "b": 50.0,
                    "c": 0.0,
                },
                expected_output=100.0,
                tags=["decision_test", "boundary"],
                description="Outer condition just barely true",
            ),
            TestCase(
                "outer_barely_false",
                {
                    "x": 5.1,
                    "t1": 5.0,
                    "y": 1.0,
                    "t2": 5.0,
                    "a": 100.0,
                    "b": 50.0,
                    "c": 0.0,
                },
                expected_output=0.0,
                tags=["decision_test", "boundary"],
                description="Outer condition just barely false",
            ),
        ]
        self.add_test_cases(cases)


class ComparisonAsValueProblem(GPProblem):
    """
    Using comparison result as a numeric value:
    f(x, y, scale) = LT(x, y) * scale

    Returns scale if x < y, else 0.
    Tests how comparison operators propagate attribution.
    """

    def __init__(self):
        super().__init__("comparison_as_value", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["x", "y", "scale"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def include_conditionals(self) -> bool:
        return True  # Need LT operator

    @property
    def ideal_expression(self) -> str:
        return "MUL(LT(x, y), scale)"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return (1.0 if inputs["x"] < inputs["y"] else 0.0) * inputs["scale"]

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "comparison_true",
                {"x": 3.0, "y": 10.0, "scale": 5.0},
                expected_output=5.0,
                tags=["comparison_test"],
                description="x < y is true, output is scale",
            ),
            TestCase(
                "comparison_false",
                {"x": 10.0, "y": 3.0, "scale": 5.0},
                expected_output=0.0,
                tags=["comparison_test"],
                description="x < y is false, output is 0",
            ),
            TestCase(
                "barely_true",
                {"x": 9.99, "y": 10.0, "scale": 100.0},
                expected_output=100.0,
                tags=["comparison_test", "boundary"],
                description="Just barely x < y",
            ),
            TestCase(
                "barely_false",
                {"x": 10.01, "y": 10.0, "scale": 100.0},
                expected_output=0.0,
                tags=["comparison_test", "boundary"],
                description="Just barely x >= y",
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Passthrough Operation Problems
# =============================================================================


class UnaryTransformProblem(GPProblem):
    """
    Chain of unary transforms: f(x) = sqrt(abs(square(x)))

    Tests that passthrough operations preserve attribution.
    All attribution should trace back to x regardless of transforms.
    """

    def __init__(self):
        super().__init__("unary_transform", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["x"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def ideal_expression(self) -> str:
        return "SQRT(ABS(SQUARE(x)))"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return math.sqrt(abs(inputs["x"] ** 2))

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "positive",
                {"x": 4.0},
                expected_output=4.0,
                tags=["passthrough_test"],
                description="x should have 100% value contribution",
            ),
            TestCase(
                "negative",
                {"x": -4.0},
                expected_output=4.0,
                tags=["passthrough_test"],
                description="Negative input, x still has 100% attribution",
            ),
            TestCase(
                "small", {"x": 0.5}, expected_output=0.5, tags=["passthrough_test"]
            ),
            TestCase(
                "zero",
                {"x": 0.0},
                expected_output=0.0,
                tags=["passthrough_test", "edge_case"],
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Polynomial Problems
# =============================================================================


class QuadraticProblem(GPProblem):
    """
    Symbolic regression: f(x) = a*x^2 + b*x + c

    Tests attribution in polynomial with multiple terms involving same variable.
    The x^2 term and x term both depend on x but contribute differently.
    """

    def __init__(self, a: float = 1.0, b: float = -2.0, c: float = 1.0):
        self.a = a
        self.b = b
        self.c = c
        super().__init__("quadratic", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["x"]

    @property
    def constants(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}

    @property
    def ideal_expression(self) -> str:
        # a*x^2 + b*x + c
        return "ADD(ADD(MUL(a, SQUARE(x)), MUL(b, x)), c)"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        x = inputs["x"]
        return self.a * x**2 + self.b * x + self.c

    def vertex_x(self) -> float:
        """X-coordinate of parabola vertex."""
        return -self.b / (2 * self.a)

    def _setup_default_test_cases(self):
        vx = self.vertex_x()
        cases = [
            TestCase(
                "at_zero",
                {"x": 0.0},
                expected_output=self.c,
                tags=["polynomial_test"],
                description="At x=0, only constant term contributes",
            ),
            TestCase(
                "at_one",
                {"x": 1.0},
                expected_output=self.a + self.b + self.c,
                tags=["polynomial_test"],
            ),
            TestCase(
                "at_vertex",
                {"x": vx},
                expected_output=self.ground_truth({"x": vx}),
                tags=["polynomial_test"],
                description=f"At vertex x={vx:.2f}, minimum/maximum of parabola",
            ),
            TestCase(
                "large_x",
                {"x": 10.0},
                expected_output=self.ground_truth({"x": 10.0}),
                tags=["polynomial_test"],
                description="Large x, quadratic term dominates",
            ),
            TestCase(
                "negative_x",
                {"x": -3.0},
                expected_output=self.ground_truth({"x": -3.0}),
                tags=["polynomial_test"],
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Min/Max Problems (Conditional-like)
# =============================================================================


class MinMaxProblem(GPProblem):
    """
    f(a, b, c) = max(a, min(b, c))

    Tests attribution through MIN/MAX which are semantically conditionals.
    """

    def __init__(self):
        super().__init__("min_max", ProblemType.SYMBOLIC_REGRESSION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["a", "b", "c"]

    @property
    def constants(self) -> Dict[str, float]:
        return {}

    @property
    def ideal_expression(self) -> str:
        return "MAX(a, MIN(b, c))"

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return max(inputs["a"], min(inputs["b"], inputs["c"]))

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "a_wins",
                {"a": 10.0, "b": 5.0, "c": 3.0},
                expected_output=10.0,
                tags=["minmax_test"],
                description="a > min(b,c), so output is a",
            ),
            TestCase(
                "min_bc_wins",
                {"a": 1.0, "b": 5.0, "c": 3.0},
                expected_output=3.0,
                tags=["minmax_test"],
                description="a < min(b,c)=c, so output is c",
            ),
            TestCase(
                "b_is_min",
                {"a": 1.0, "b": 2.0, "c": 5.0},
                expected_output=2.0,
                tags=["minmax_test"],
                description="min(b,c)=b, and b > a",
            ),
            TestCase(
                "all_equal",
                {"a": 5.0, "b": 5.0, "c": 5.0},
                expected_output=5.0,
                tags=["minmax_test", "edge_case"],
                description="All equal",
            ),
            TestCase(
                "a_equals_min",
                {"a": 3.0, "b": 5.0, "c": 3.0},
                expected_output=3.0,
                tags=["minmax_test", "boundary"],
                description="a equals min(b,c) - boundary case",
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Complex Real-World-Like Problems
# =============================================================================


class CreditScoreProblem(GPProblem):
    """
    Simplified credit scoring model (inspired by real use cases):

    score = base + income_factor - debt_factor + age_bonus
    approved = score > threshold

    Tests a more realistic scenario with multiple factors.
    """

    def __init__(self):
        super().__init__("credit_score", ProblemType.BINARY_CLASSIFICATION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["income", "debt", "age", "threshold"]

    @property
    def constants(self) -> Dict[str, float]:
        return {
            "BASE": 500.0,
            "INCOME_WEIGHT": 0.01,  # $100k income adds 1000 to score
            "DEBT_WEIGHT": 0.02,  # $50k debt subtracts 1000
            "AGE_BONUS": 5.0,  # Each year adds 5 points
            "ONE": 1.0,
            "ZERO": 0.0,
        }

    @property
    def ideal_expression(self) -> str:
        # score = BASE + income*0.01 - debt*0.02 + age*5
        # IFLTE(threshold, score, ONE, ZERO)  -- approved if threshold <= score
        score = "ADD(ADD(ADD(BASE, MUL(INCOME_WEIGHT, income)), NEG(MUL(DEBT_WEIGHT, debt))), MUL(AGE_BONUS, age))"
        return f"IFLTE(threshold, {score}, ONE, ZERO)"

    def compute_score(self, inputs: Dict[str, float]) -> float:
        return (
            500.0
            + 0.01 * inputs["income"]
            - 0.02 * inputs["debt"]
            + 5.0 * inputs["age"]
        )

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        score = self.compute_score(inputs)
        return 1.0 if inputs["threshold"] <= score else 0.0

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "high_income_approved",
                {"income": 100000.0, "debt": 10000.0, "age": 35.0, "threshold": 700.0},
                expected_class=1,
                tags=["real_world"],
                description="High income ($100k), low debt → approved",
            ),
            TestCase(
                "high_debt_denied",
                {"income": 50000.0, "debt": 80000.0, "age": 25.0, "threshold": 700.0},
                expected_class=0,
                tags=["real_world"],
                description="Moderate income but high debt → denied",
            ),
            TestCase(
                "young_low_income",
                {"income": 30000.0, "debt": 5000.0, "age": 22.0, "threshold": 600.0},
                expected_class=1,
                tags=["real_world"],
                description="Young, low income but also low debt",
            ),
            TestCase(
                "borderline",
                {"income": 60000.0, "debt": 30000.0, "age": 30.0, "threshold": 710.0},
                expected_class=0,
                tags=["real_world", "boundary"],
                description="Score ~700, threshold 710 → just denied",
            ),
            TestCase(
                "age_compensates",
                {"income": 40000.0, "debt": 20000.0, "age": 55.0, "threshold": 650.0},
                expected_class=1,
                tags=["real_world", "attribution_test"],
                description="Lower income but age bonus helps",
            ),
        ]
        self.add_test_cases(cases)


class DistanceClassificationProblem(GPProblem):
    """
    3D distance-based classification:
    Class 1 if point is within radius of center, else Class 0.

    Extends PointInCircle to 3D to test scaling.
    """

    def __init__(self):
        super().__init__("distance_3d", ProblemType.BINARY_CLASSIFICATION)
        self._setup_default_test_cases()

    @property
    def input_terminals(self) -> List[str]:
        return ["cx", "cy", "cz", "px", "py", "pz", "r"]

    @property
    def constants(self) -> Dict[str, float]:
        return {"ZERO": 0.0, "ONE": 1.0}

    @property
    def ideal_expression(self) -> str:
        # distance = sqrt((cx-px)^2 + (cy-py)^2 + (cz-pz)^2)
        dist = "SQRT(ADD(ADD(SQUARE(SUB(cx, px)), SQUARE(SUB(cy, py))), SQUARE(SUB(cz, pz))))"
        return f"IFLTE({dist}, r, ONE, ZERO)"

    def compute_distance(self, inputs: Dict[str, float]) -> float:
        return math.sqrt(
            (inputs["cx"] - inputs["px"]) ** 2
            + (inputs["cy"] - inputs["py"]) ** 2
            + (inputs["cz"] - inputs["pz"]) ** 2
        )

    def ground_truth(self, inputs: Dict[str, float]) -> float:
        return 1.0 if self.compute_distance(inputs) <= inputs["r"] else 0.0

    def _setup_default_test_cases(self):
        cases = [
            TestCase(
                "inside_origin",
                {
                    "cx": 0.0,
                    "cy": 0.0,
                    "cz": 0.0,
                    "px": 1.0,
                    "py": 1.0,
                    "pz": 1.0,
                    "r": 3.0,
                },
                expected_class=1,
                tags=["3d_test"],
                description="Point (1,1,1) within r=3 of origin",
            ),
            TestCase(
                "outside",
                {
                    "cx": 0.0,
                    "cy": 0.0,
                    "cz": 0.0,
                    "px": 5.0,
                    "py": 5.0,
                    "pz": 5.0,
                    "r": 3.0,
                },
                expected_class=0,
                tags=["3d_test"],
                description="Point (5,5,5) outside r=3 sphere",
            ),
            TestCase(
                "z_dominant",
                {
                    "cx": 0.0,
                    "cy": 0.0,
                    "cz": 0.0,
                    "px": 0.1,
                    "py": 0.1,
                    "pz": 4.0,
                    "r": 5.0,
                },
                expected_class=1,
                tags=["3d_test", "attribution_test"],
                description="Z-displacement dominates distance",
            ),
            TestCase(
                "on_surface",
                {
                    "cx": 0.0,
                    "cy": 0.0,
                    "cz": 0.0,
                    "px": 3.0,
                    "py": 0.0,
                    "pz": 0.0,
                    "r": 3.0,
                },
                expected_class=1,
                tags=["3d_test", "boundary"],
                description="Exactly on sphere surface",
            ),
        ]
        self.add_test_cases(cases)


# =============================================================================
# Utility Functions
# =============================================================================


def rank_correlation(contrib1: Dict[str, float], contrib2: Dict[str, float]) -> float:
    """Spearman rank correlation between two attribution dicts."""
    common = {k for k in set(contrib1) & set(contrib2) if not k.startswith("_")}
    if len(common) < 2:
        return float("nan")

    sorted1 = sorted(common, key=lambda x: -contrib1[x])
    sorted2 = sorted(common, key=lambda x: -contrib2[x])
    rank1 = {t: i for i, t in enumerate(sorted1)}
    rank2 = {t: i for i, t in enumerate(sorted2)}

    n = len(common)
    d_squared = sum((rank1[t] - rank2[t]) ** 2 for t in common)
    return 1 - (6 * d_squared) / (n * (n**2 - 1))


def show_all(
    results: TestSuiteResults,
    show_tree_per_result: bool = False,
    show_counterfactuals: bool = True,
    max_results: Optional[int] = None,
):
    """
    Display all results in Jupyter.

    Args:
        results: TestSuiteResults to display
        show_tree_per_result: If False, only show tree in summary (default)
        show_counterfactuals: Whether to show counterfactual results
        max_results: Maximum number of individual results to show (None = all)
    """
    if HAS_IPYTHON:
        display(results)

        results_to_show = results.results
        if max_results is not None:
            results_to_show = results_to_show[:max_results]

        for r in results_to_show:
            if not show_tree_per_result:
                # Temporarily hide tree expression for display
                original_tree = r.tree_expression
                r.tree_expression = None
                display(r)
                r.tree_expression = original_tree
            else:
                display(r)

        if max_results is not None and len(results.results) > max_results:
            display(
                HTML(
                    f"<i style='color: #666;'>... and {len(results.results) - max_results} more results</i>"
                )
            )

        if show_counterfactuals:
            for cf in results.counterfactual_results:
                display(cf)
    else:
        print(results)


# =============================================================================
# Problem Registry
# =============================================================================

ALL_PROBLEMS = {
    # Basic arithmetic
    "weighted_sum": WeightedSumProblem,
    "multiplicative": MultiplicativeProblem,
    "mixed_arithmetic": MixedArithmeticProblem,
    "quadratic": QuadraticProblem,
    "unary_transform": UnaryTransformProblem,
    # Conditional/branching
    "simple_conditional": SimpleConditionalProblem,
    "nested_conditional": NestedConditionalProblem,
    "comparison_as_value": ComparisonAsValueProblem,
    "min_max": MinMaxProblem,
    # Classification
    "point_in_circle": PointInCircleProblem,
    "distance_3d": DistanceClassificationProblem,
    # Real-world inspired
    "credit_score": CreditScoreProblem,
}


def list_problems() -> None:
    """Print a summary of all available test problems."""
    print("Available FCP Test Problems")
    print("=" * 60)

    categories = {
        "Additive Attribution": ["weighted_sum"],
        "Multiplicative Attribution (log-space)": [
            "multiplicative",
            "mixed_arithmetic",
        ],
        "Passthrough Operations": ["unary_transform"],
        "Polynomial": ["quadratic"],
        "Conditional (dual-channel)": [
            "simple_conditional",
            "nested_conditional",
            "comparison_as_value",
        ],
        "Min/Max": ["min_max"],
        "Distance-based Classification": ["point_in_circle", "distance_3d"],
        "Real-world Inspired": ["credit_score"],
    }

    for category, problem_names in categories.items():
        print(f"\n{category}:")
        for name in problem_names:
            prob_class = ALL_PROBLEMS[name]
            prob = prob_class()
            print(f"  • {name}: {len(prob.test_cases)} test cases")
            print(f"    {prob_class.__doc__.strip().split(chr(10))[0]}")


def create_all_problems() -> Dict[str, GPProblem]:
    """Instantiate all available problems."""
    return {name: cls() for name, cls in ALL_PROBLEMS.items()}


def run_all_problems(
    runner: TestRunner,
    create_pset_fn: Callable,
    tracked_value_class: Any,
    gp_module: Any,
    verbose: bool = False,
) -> Dict[str, TestSuiteResults]:
    """
    Run all problems and return results.

    Args:
        runner: TestRunner instance
        create_pset_fn: Your create_tracked_pset function
        tracked_value_class: Your TrackedValue class
        gp_module: DEAP gp module
        verbose: Whether to print progress

    Returns:
        Dict mapping problem name to TestSuiteResults
    """
    all_results = {}

    for name, prob_class in ALL_PROBLEMS.items():
        if verbose:
            print(f"Running {name}...")

        problem = prob_class()
        try:
            pset, tree = problem.setup(create_pset_fn, tracked_value_class, gp_module)
            results = runner.run(problem)
            all_results[name] = results

            if verbose:
                acc = results.accuracy
                mae = results.mean_absolute_error
                if acc is not None:
                    print(f"  Accuracy: {acc:.1%}")
                if mae is not None:
                    print(f"  MAE: {mae:.4f}")
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            all_results[name] = None

    return all_results
