"""
FCP Waterfall Plot

Generates a SHAP-style waterfall chart showing how FCP signed contributions
progressively build from the baseline to the final prediction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import numpy as np
from deap import gp
from typing import Dict, List, Optional, Tuple


def plot_fcp_waterfall(
    signed_contributions,
    output_value: float,
    feature_names: List[str],
    feature_values: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    max_features: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    positive_color: str = "#ff0051",
    negative_color: str = "#008bfb",
    baseline_color: str = "#999999",
    bar_height: float = 0.55,
    show_feature_values: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot FCP signed contributions as a waterfall chart.

    Starts at the baseline value and progressively adds each feature's
    signed contribution to arrive at the final output — directly
    comparable to SHAP's force/waterfall plots.

    Args:
        signed_contributions: SignedContributions object from FCP evaluation
        output_value: The model's output for this instance
        feature_names: List of all feature names
        feature_values: Optional dict of feature name -> value for annotation
        title: Optional plot title
        max_features: Maximum features to show (rest grouped as "other")
        figsize: Figure size tuple
        positive_color: Color for positive contributions
        negative_color: Color for negative contributions
        baseline_color: Color for baseline bar
        bar_height: Height of bars
        show_feature_values: Whether to show feature values on y-axis
        ax: Optional matplotlib axes to draw on

    Returns:
        matplotlib Figure
    """
    # Extract contributions
    contribs = {}
    for name in feature_names:
        val = signed_contributions.contributions.get(name, 0.0)
        contribs[name] = val

    baseline = signed_contributions.baseline

    # Sort by absolute magnitude descending
    sorted_items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Group small contributions if needed
    if len(sorted_items) > max_features:
        shown = sorted_items[:max_features]
        rest_val = sum(v for _, v in sorted_items[max_features:])
        if abs(rest_val) > 1e-6:
            shown.append(("other features", rest_val))
    else:
        shown = sorted_items

    # Filter out zero contributions for cleaner display
    shown = [(name, val) for name, val in shown if abs(val) > 1e-4]

    # Build waterfall data: baseline at top, features stacked down
    n_bars = len(shown) + 1  # +1 for baseline only (output shown as vertical line)

    labels = ["baseline"] + [name for name, _ in shown]
    values = [baseline] + [val for _, val in shown]

    # Compute cumulative positions for the waterfall
    starts = [0.0]  # baseline starts at 0
    running = baseline
    for _, val in shown:
        starts.append(running)
        running += val

    # Bar widths
    widths = [baseline] + [val for _, val in shown]

    # Y positions (reversed so baseline is at top, last feature at bottom)
    y_positions = list(range(n_bars - 1, -1, -1))

    # Create figure
    if ax is None:
        if figsize is None:
            figsize = (9, max(3.5, 0.5 * n_bars + 1))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine x-axis range for consistent layout
    cumulative = [baseline]
    running = baseline
    for _, val in shown:
        running += val
        cumulative.append(running)
    all_values = cumulative + [0, output_value, baseline]
    x_min = min(all_values) - abs(output_value) * 0.15
    x_max = max(all_values) + abs(output_value) * 0.15

    # Draw connector lines (thin gray lines connecting bar ends)
    running = baseline
    for i in range(len(shown)):
        y_top = y_positions[i + 1]  # current feature bar
        y_bot = y_positions[i + 2] if i + 2 < len(y_positions) else None  # next bar

        end_x = running + shown[i][1]

        # Connector from this bar's end to the next bar
        if y_bot is not None:
            ax.plot(
                [end_x, end_x],
                [y_top - bar_height / 2, y_bot + bar_height / 2],
                color="#cccccc",
                linewidth=0.8,
                zorder=1,
            )

        running = end_x

    # Connect baseline to first feature
    if shown:
        ax.plot(
            [baseline, baseline],
            [y_positions[0] - bar_height / 2, y_positions[1] + bar_height / 2],
            color="#cccccc",
            linewidth=0.8,
            zorder=1,
        )

    # Draw bars
    for i, (label, width) in enumerate(zip(labels, widths)):
        y = y_positions[i]
        start = starts[i]

        if i == 0:
            # Baseline bar
            color = baseline_color
            alpha = 0.35
            edgecolor = baseline_color
        else:
            # Feature bar
            color = positive_color if width >= 0 else negative_color
            alpha = 0.85
            edgecolor = "white"

        ax.barh(
            y,
            width,
            left=start,
            height=bar_height,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=0.5,
            zorder=2,
        )

        # Value annotation
        if i == 0:
            # Baseline: show value at right end
            ax.text(
                baseline + 0.2,
                y,
                f"{baseline:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color=baseline_color,
                fontweight="bold",
            )
        else:
            # Feature: show signed value
            text_x = start + width
            if width >= 0:
                text_x += 0.15
                ha = "left"
            else:
                text_x -= 0.15
                ha = "right"

            ax.text(
                text_x,
                y,
                f"{width:+.2f}",
                va="center",
                ha=ha,
                fontsize=9,
                color=color,
                fontweight="bold",
                alpha=1.0,
            )

    # Y-axis labels
    y_labels = []
    for i, label in enumerate(labels):
        if i == 0 or i == n_bars - 1:
            y_labels.append(label)
        else:
            feature_name = label
            if (
                show_feature_values
                and feature_values
                and feature_name in feature_values
            ):
                fv = feature_values[feature_name]
                if fv == int(fv):
                    y_labels.append(f"{feature_name} = {int(fv)}")
                else:
                    y_labels.append(f"{feature_name} = {fv:.2f}")
            else:
                y_labels.append(feature_name)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    # Style
    ax.set_xlim(x_min, x_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlabel("Model output value", fontsize=9, color="#666666")

    # f(x) output line — vertical line spanning the full plot with label at top
    ax.axvline(
        x=output_value,
        color="#333333",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        zorder=3,
    )
    ax.text(
        output_value,
        max(y_positions) + 0.7,
        f"f(x) = {output_value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def plot_fcp_vs_shap(
    fcp_signed_contributions,
    shap_values: np.ndarray,
    shap_base_value: float,
    output_value: float,
    feature_names: List[str],
    feature_values: Optional[Dict[str, float]] = None,
    max_features: int = 8,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of FCP and SHAP waterfall plots.

    Args:
        fcp_signed_contributions: SignedContributions from FCP
        shap_values: SHAP values array for this instance
        shap_base_value: SHAP expected value
        output_value: Model output
        feature_names: Feature name list
        feature_values: Optional feature value dict
        max_features: Max features to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from fcp_tracker import SignedContributions

    if figsize is None:
        n_features = min(len(feature_names), max_features)
        figsize = (16, max(4, 0.5 * (n_features + 2) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # FCP waterfall
    plot_fcp_waterfall(
        fcp_signed_contributions,
        output_value,
        feature_names,
        feature_values=feature_values,
        title="FCP Signed Contributions",
        max_features=max_features,
        ax=ax1,
    )

    # Build SHAP SignedContributions-like object for reuse
    shap_contribs = SignedContributions(baseline=shap_base_value)
    for i, name in enumerate(feature_names):
        if i < len(shap_values):
            shap_contribs.contributions[name] = float(shap_values[i])

    plot_fcp_waterfall(
        shap_contribs,
        output_value,
        feature_names,
        feature_values=feature_values,
        title="SHAP Values",
        max_features=max_features,
        ax=ax2,
    )

    # Sync x-axis limits so the f(x) vertical lines align visually
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    fig.tight_layout()
    return fig


def make_fcp_shap_comparison(results, X_test, info, sample_count=200):
    func = gp.compile(results["best_tree"], results["pset"])

    def predict(X):
        return np.array([func(*xi) for xi in X])

    background = shap.sample(X_test, sample_count, random_state=42)
    explainer = shap.KernelExplainer(predict, background, algorithm="auto")
    shap_values = explainer.shap_values(X_test)
    shap.initjs()

    figs = []
    for i, instance in enumerate(results["explanations"]):
        # Get the FCP explanation for the same instance
        explanation = instance
        sc = explanation["result"].output.signed_contributions

        # Feature values for annotation
        instance_values = {name: float(val) for name, val in zip(info.names, X_test[i])}

        fig = plot_fcp_vs_shap(
            fcp_signed_contributions=sc,
            shap_values=shap_values[i],
            shap_base_value=explainer.expected_value,
            output_value=explanation["score"],
            feature_names=info.names,
            feature_values=instance_values,
        )
        figs.append(fig)
    return figs
