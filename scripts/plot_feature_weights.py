"""
plot_feature_weights.py — Chart the top feature importances for the three
primary probability heads: p_wr (WR/Serious), p_injury, p_pen (P95 penalty).

Usage:
    python scripts/plot_feature_weights.py
    python scripts/plot_feature_weights.py --top 15
    python scripts/plot_feature_weights.py --output plots/feature_weights.png

The chart is saved to plots/feature_weights.png (or --output path) and
also displayed interactively if a display is available.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to TkAgg if you want a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CACHE_DIR = "ml_cache"

# Feature display names from ml_risk_scorer (duplicated here for standalone use)
FEATURE_DISPLAY = {
    "log_inspections":          "Inspection Count (log)",
    "log_violations":           "Violation Count (log)",
    "serious_violations":       "Serious Violations",
    "willful_violations":       "Willful Violations",
    "repeat_violations":        "Repeat Violations",
    "log_penalties":            "Total Penalties (log $)",
    "avg_penalty":              "Avg Penalty ($)",
    "max_penalty":              "Max Single Penalty ($)",
    "recent_ratio":             "Recent Activity (1yr)",
    "severe_incidents":         "Fat/Cat Inspections",
    "violations_per_inspection":"Violations / Inspection",
    "accident_count":           "Linked Accidents",
    "fatality_count":           "Fatalities",
    "injury_count":             "Reported Injuries",
    "avg_gravity":              "Avg Violation Gravity",
    "penalties_per_inspection": "Penalties / Inspection ($)",
    "clean_ratio":              "Clean Inspection Ratio",
    "time_adjusted_penalty":    "Time-Adjusted Penalty ($)",
    "recent_wr_rate":           "Recent W/R Rate",
    "trend_delta":              "Violation Trend (Δ vpi)",
    "relative_violation_rate":  "Violation Rate vs. Industry (z)",
    "relative_penalty":         "Avg Penalty vs. Industry (z)",
    "relative_serious_ratio":   "Serious Ratio vs. Industry (z)",
    "relative_willful_repeat":  "Willful+Repeat Rate vs. Industry (z)",
}


def _get_importances(head, feature_names):
    """Extract a {feature: importance} dict from an HGBC or fallback object.

    sklearn 1.6+ removed the feature_importances_ property from
    HistGradientBoostingClassifier.  We compute gain-based importances
    directly from the internal tree predictor nodes (each non-leaf node
    has a 'gain' and 'feature_idx' field).
    """
    # Fast path: property still available (sklearn < 1.6)
    fi = getattr(head, "feature_importances_", None)
    if fi is not None:
        if len(fi) != len(feature_names):
            min_len = min(len(fi), len(feature_names))
            fi = fi[:min_len]
            feature_names = feature_names[:min_len]
        return dict(zip(feature_names, fi.tolist()))

    # Slow path: sum gains per feature across all trees
    predictors = getattr(head, "_predictors", None)
    n_features = getattr(head, "n_features_in_", None)
    if predictors is None or n_features is None:
        return None

    gains = np.zeros(n_features, dtype=float)
    for stage in predictors:            # one stage per boosting round
        for tree in stage:              # one tree per class (1 for binary)
            nodes = tree.nodes
            # non-leaf nodes have gain > 0 and a valid feature_idx
            mask = nodes["is_leaf"] == 0
            for node in nodes[mask]:
                fi_idx = int(node["feature_idx"])
                if 0 <= fi_idx < n_features:
                    gains[fi_idx] += float(node["gain"])

    total = gains.sum()
    if total == 0:
        return None
    gains /= total

    # Align with the feature_names list (may be shorter due to NAICS one-hots)
    n = min(len(gains), len(feature_names))
    return dict(zip(feature_names[:n], gains[:n].tolist()))


def _top_n(importance_dict, n, exclude_naics=True):
    """Return top-n (name, importance) pairs, optionally dropping NAICS one-hots."""
    items = [
        (k, v) for k, v in importance_dict.items()
        if not (exclude_naics and k.startswith("naics_"))
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]


def plot_feature_weights(top_n: int = 12, output_path: str = "plots/feature_weights.png"):
    from src.scoring.ml_risk_scorer import MLRiskScorer
    from src.scoring.multi_target_scorer import MultiTargetRiskScorer, MODEL_FILE

    model_path = os.path.join(CACHE_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run 'python scripts/train_multi_target.py' first.")
        sys.exit(1)

    print(f"Loading model from {model_path} …")
    mt = MultiTargetRiskScorer.load(model_path)
    if not mt.is_fitted:
        print("ERROR: Model loaded but is_fitted=False.")
        sys.exit(1)

    # Load feature names from the base scorer
    scorer = MLRiskScorer()
    feature_names = scorer.FEATURE_NAMES

    heads = [
        ("p_wr\n(WR / Serious Event)",        mt._head_wr,       "#2166ac"),
        ("p_injury\n(Hospitalization / Fatal)", mt._head_injury,   "#d6604d"),
        ("p_pen\n(Penalty ≥ Industry P95)",     mt._head_pen_p95,  "#4dac26"),
    ]

    # ── Extract importances ────────────────────────────────────────────────
    all_data = []
    for label, head, color in heads:
        if head is None:
            print(f"  WARNING: head for {label!r} is None — skipping.")
            all_data.append(None)
            continue
        imp = _get_importances(head, feature_names)
        if imp is None:
            print(f"  WARNING: {label!r} has no feature_importances_ (possibly _ConstantClassifier).")
            all_data.append(None)
        else:
            top = _top_n(imp, top_n, exclude_naics=True)
            all_data.append(top)
            print(f"  {label.split(chr(10))[0]}: top feature = {top[0][0]} ({top[0][1]:.4f})")

    # ── Build figure ───────────────────────────────────────────────────────
    valid = [(label, data, color) for (label, _, color), data in zip(heads, all_data) if data]
    n_plots = len(valid)
    if n_plots == 0:
        print("ERROR: No feature importances could be extracted.")
        sys.exit(1)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        "Top Feature Importances by Probability Head\n(MDI — excluding NAICS industry one-hots)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (label, top_items, color) in zip(axes, valid):
        names  = [FEATURE_DISPLAY.get(n, n) for n, _ in top_items]
        values = [v for _, v in top_items]

        # Normalise so the bars sum to 1 within each panel for easier comparison
        total = sum(values) or 1.0
        values_norm = [v / total for v in values]

        # Horizontal bars — largest at top
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values_norm, color=color, alpha=0.82, edgecolor="white", linewidth=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()   # largest at top
        ax.set_xlabel("Relative importance (normalised within head)", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=8)

        # Value labels on bars
        for bar, val in zip(bars, values_norm):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", ha="left", fontsize=7.5, color="#333333",
            )

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {output_path}")

    # Try to show interactively (silently skip if no display)
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Plot feature weights for the three probability heads")
    parser.add_argument("--top",    type=int, default=12,
                        help="Number of top features to show per head (default 12)")
    parser.add_argument("--output", type=str, default="plots/feature_weights.png",
                        help="Output image path (default plots/feature_weights.png)")
    args = parser.parse_args()
    plot_feature_weights(top_n=args.top, output_path=args.output)


if __name__ == "__main__":
    main()
