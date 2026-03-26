"""
tests/validation_plots.py

Generates diagnostic matplotlib plots from a RealWorldData instance and
saves them to plots/ under the project root.

Plots produced
--------------
  01_scatter_score_vs_adverse.png   -- Score vs future adverse outcome with
                                       rolling-mean smooth + faint linear fit
  02_binned_mean_outcome.png        -- Mean outcome per quantile bin + 95% CI bars
  03_decile_lift.png                -- Decile lift bar chart (lift + n annotations)
  04_capture_curve.png              -- Top-K cumulative event capture curve
  05_distribution_overlay.png      -- Score distribution: all vs. high-future-risk
  06_decile_boxplot.png             -- Boxplot of adverse outcome by score decile

Console output
--------------
  Cumulative gains table (top 5 / 10 / 20 / 30 %)

Usage (standalone)
------------------
    python tests/validation_plots.py

Usage (from code)
-----------------
    from tests.validation_plots import generate_all_validation_plots
    from tests.test_real_world_validation import RealWorldData
    generate_all_validation_plots(RealWorldData.get())
"""

import os
import sys

# Allow project-root imports (src/, etc.) regardless of working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TESTS_DIR    = os.path.dirname(__file__)
for _p in (_PROJECT_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math
import numpy as np
import pandas as pd

# Use the non-interactive Agg backend so the module is safe in headless /
# CI contexts.  Must be set *before* importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import helpers from the validation test module.
from test_real_world_validation import (  # noqa: E402
    RealWorldData,
    _decile_summary,
    _compute_threshold_metrics,
    _compute_topk_capture,
    EVAL_THRESHOLDS,
    TOPK_FRACTIONS,
)
# Note: _rolling_smooth is defined in this file (validation_plots.py).

# -- Output directory -------------------------------------------------
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")

# -- Visual style constants -------------------------------------------
FONT_TITLE  = 15
FONT_LABEL  = 13
FONT_TICK   = 11
FONT_ANNOT  = 9
FONT_LEGEND = 11
FIG_SIZE    = (9, 6)
FIG_DPI     = 150
COLOR_MAIN  = "#2c7bb6"   # steel blue
COLOR_ALT   = "#d7191c"   # tomato red
COLOR_REF   = "#aaaaaa"   # grey reference lines
COLOR_CI    = "#abd9e9"   # light blue for CI shading


# ====================================================================== #
#  Statistical helpers
# ====================================================================== #

def _bootstrap_mean_ci_95(
    values: np.ndarray,
    n_boot: int = 500,
    rng: np.random.Generator = None,
) -> float:
    """Bootstrap half-width of the 95% CI for the mean.

    Falls back to 0 when n <= 1.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(42)
    boot_means = np.array([
        values[rng.integers(0, n, size=n)].mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (hi - lo) / 2.0


def _quantile_bin_stats(
    scores: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Split scores into n_bins quantile bins (equal sample count).

    Returns a DataFrame with columns:
        bin, score_lo, score_hi, score_mid, n, mean, ci_hw
    where ci_hw is the bootstrap 95% CI half-width for the mean.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(scores, percentiles)
    rng   = np.random.default_rng(42)

    rows = []
    for i in range(n_bins):
        lo   = edges[i]
        hi   = edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 \
               else (scores >= lo) & (scores < hi)
        vals = outcomes[mask]
        n    = int(mask.sum())
        mean = float(vals.mean()) if n > 0 else float("nan")
        ci   = _bootstrap_mean_ci_95(vals, rng=rng) if n > 1 else 0.0
        rows.append({
            "bin":       i + 1,
            "score_lo":  round(float(lo), 2),
            "score_hi":  round(float(hi), 2),
            "score_mid": round((float(lo) + float(hi)) / 2.0, 2),
            "n":         n,
            "mean":      mean,
            "ci_hw":     ci,
        })
    return pd.DataFrame(rows)


def _rolling_smooth(
    scores: np.ndarray,
    outcomes: np.ndarray,
    window_frac: float = 0.08,
    min_window: int    = 50,
) -> tuple:
    """Return a centred rolling-mean smooth of outcomes sorted by score.

    Approximates LOWESS behaviour without external dependencies.

    Args:
        scores       : predictor array
        outcomes     : target array (parallel to scores)
        window_frac  : rolling window as a fraction of total n
        min_window   : minimum window size in samples

    Returns:
        xs     : sorted score values
        ys_sm  : corresponding smoothed outcome values
    """
    sort_idx = np.argsort(scores)
    xs       = scores[sort_idx]
    ys       = outcomes[sort_idx]
    win      = max(min_window, int(len(xs) * window_frac))
    ys_sm    = (
        pd.Series(ys)
        .rolling(window=win, center=True, min_periods=max(1, win // 4))
        .mean()
        .values
    )
    return xs, ys_sm


# ====================================================================== #
#  Internal plot functions
# ====================================================================== #

def _plot_scatter_score_vs_adverse(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 1 -- scatter of baseline score vs future adverse-outcome score.

    Overlays:
      - raw scatter (alpha=0.3)
      - LOWESS-style rolling-mean smooth (primary trend, bold)
      - linear fit from np.polyfit (secondary, faint dashed)
    """
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Raw scatter
    ax.scatter(scores, adverse, alpha=0.3, s=6, color=COLOR_MAIN,
               rasterized=True, label="_nolegend_")

    # Rolling-mean smooth (LOWESS-style)
    xs_sm, ys_sm = _rolling_smooth(scores, adverse)
    ax.plot(xs_sm, ys_sm, color=COLOR_ALT, linewidth=2.5,
            label="Rolling mean (trend)")

    # Faint linear fit
    m, b   = np.polyfit(scores, adverse, 1)
    x_line = np.linspace(float(scores.min()), float(scores.max()), 300)
    ax.plot(
        x_line, m * x_line + b,
        color="#555555", linewidth=1.2, linestyle="--",
        label=f"Linear fit  (slope={m:.4f})",
        alpha=0.6,
    )

    ax.set_xlabel("Baseline Risk Score", fontsize=FONT_LABEL)
    ax.set_ylabel("Future Adverse Outcome Score", fontsize=FONT_LABEL)
    ax.set_title(
        "Baseline Risk Score vs Future Adverse Outcome",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "01_scatter_score_vs_adverse.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_binned_mean_outcome(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 2 -- mean future adverse outcome per score quantile bin.

    10 quantile bins ensure roughly equal sample counts per bin, making
    the error bars directly comparable across the score range.  95% CI
    half-widths are computed via bootstrap.
    """
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    df = _quantile_bin_stats(scores, adverse, n_bins=10)
    valid = df["mean"].notna()

    xs    = df.loc[valid, "score_mid"].values
    means = df.loc[valid, "mean"].values
    ci_hw = df.loc[valid, "ci_hw"].values
    ns    = df.loc[valid, "n"].values
    los   = df.loc[valid, "score_lo"].values
    his   = df.loc[valid, "score_hi"].values

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # CI band
    ax.fill_between(
        xs, means - ci_hw, means + ci_hw,
        alpha=0.25, color=COLOR_MAIN, label="95% CI (bootstrap)",
    )

    # Mean line with markers
    ax.plot(
        xs, means,
        marker="o", markersize=8, linewidth=2.2,
        color=COLOR_MAIN, label="Mean adverse outcome",
    )

    # Error bars
    ax.errorbar(
        xs, means, yerr=ci_hw,
        fmt="none", ecolor=COLOR_MAIN, elinewidth=1.5, capsize=4,
    )

    # Annotate each point with n and score range
    for cx, my, ci, n, lo, hi in zip(xs, means, ci_hw, ns, los, his):
        ax.annotate(
            f"n={n:,}\n[{lo:.0f},{hi:.0f}]",
            xy=(cx, my + ci),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=FONT_ANNOT,
            ha="center",
            color="#333333",
        )

    ax.set_xlabel("Score Quantile Bin (mid-point shown)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean Future Adverse Outcome Score", fontsize=FONT_LABEL)
    ax.set_title(
        "Mean Future Adverse Outcome by Score Quantile",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "02_binned_mean_outcome.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_decile_lift(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 3 -- decile lift bar chart (uses _decile_summary).

    Bars above 1.0 (model beats random) are coloured blue; bars below are red.
    Each bar is annotated with lift value and sample count.
    """
    df = _decile_summary(
        rw_data.paired_scores,
        rw_data.paired_adverse_scores,
        outcome_label="adverse_score",
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bar_colors = [COLOR_MAIN if lft >= 1.0 else COLOR_ALT for lft in df["lift"]]
    bars = ax.bar(df["decile"], df["lift"], color=bar_colors, edgecolor="white", width=0.7)

    y_max = float(df["lift"].max())

    # Annotate each bar with lift value (above bar) and n (inside bar)
    for bar, lft, n in zip(bars, df["lift"], df["n"]):
        bx = bar.get_x() + bar.get_width() / 2
        bh = bar.get_height()
        # Lift label above the bar
        ax.text(
            bx, bh + y_max * 0.015,
            f"{lft:.2f}x",
            ha="center", va="bottom",
            fontsize=FONT_ANNOT, fontweight="bold", color="#111111",
        )
        # n inside the bar (only when bar is tall enough)
        if bh > y_max * 0.08:
            ax.text(
                bx, bh * 0.35,
                f"n={n:,}",
                ha="center", va="center",
                fontsize=FONT_ANNOT - 1, color="white", fontweight="bold",
            )

    ax.axhline(y=1.0, color=COLOR_REF, linewidth=2, linestyle="--",
               label="Random baseline (lift=1.0)")

    ax.set_xlabel("Decile  (1 = lowest score  ->  10 = highest score)",
                  fontsize=FONT_LABEL)
    ax.set_ylabel("Lift  (decile mean / population mean)", fontsize=FONT_LABEL)
    ax.set_title("Decile Lift (Risk Concentration)", fontsize=FONT_TITLE,
                 fontweight="bold")
    ax.set_xticks(df["decile"])
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, axis="y", linewidth=0.5)
    ax.set_ylim(0, y_max * 1.15)

    fig.tight_layout()
    path = os.path.join(plots_dir, "03_decile_lift.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_capture_curve(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 4 -- cumulative adverse-outcome capture curve (gain curve).

    Establishments are sorted by baseline score (highest first).  The curve
    shows what fraction of the total future adverse-outcome score is captured
    by inspecting the top-k% of establishments.
    """
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    order          = np.argsort(scores)[::-1]
    sorted_adverse = adverse[order]

    n             = len(sorted_adverse)
    total_adverse = float(sorted_adverse.sum())
    if total_adverse == 0.0:
        total_adverse = 1.0

    pct_pop     = np.concatenate([[0.0], np.arange(1, n + 1) / n * 100.0])
    pct_adverse = np.concatenate([[0.0], np.cumsum(sorted_adverse) / total_adverse * 100.0])

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(pct_pop, pct_adverse, color=COLOR_MAIN, linewidth=2.5, label="Model")
    ax.plot([0, 100], [0, 100], color=COLOR_REF, linewidth=1.5, linestyle="--",
            label="Random baseline")

    # Annotate top-10% capture rate
    idx_10 = int(round(n * 0.10))
    cap_10 = float(sorted_adverse[:max(idx_10, 1)].sum()) / total_adverse * 100.0
    ax.annotate(
        f"Top 10%\ncaptures {cap_10:.1f}%",
        xy=(10, cap_10),
        xytext=(18, cap_10 - 8),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3),
        fontsize=FONT_ANNOT + 1,
        color="#222222",
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of Establishments  (sorted by score, highest first)",
                  fontsize=FONT_LABEL)
    ax.set_ylabel("Cumulative % of Total Adverse Outcome Captured",
                  fontsize=FONT_LABEL)
    ax.set_title("Top-K Event Capture Curve", fontsize=FONT_TITLE,
                 fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "04_capture_curve.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_distribution_overlay(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 5 -- baseline score distributions for all vs. high-future-risk subgroup.

    The "high future adverse" group is establishments whose future adverse-outcome
    score falls in the top 25% of the paired population.  density=True makes the
    two histograms directly comparable despite different group sizes.
    """
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    threshold   = float(np.percentile(adverse, 75))
    high_mask   = adverse >= threshold
    high_scores = scores[high_mask]
    n_high      = int(high_mask.sum())
    n_all       = len(scores)

    score_max = float(scores.max())
    bin_edges = np.linspace(0.0, max(score_max + 1.0, 1.0), 41)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.hist(
        scores, bins=bin_edges,
        alpha=0.5, color=COLOR_MAIN, density=True,
        label=f"All establishments  (n={n_all:,})",
    )
    ax.hist(
        high_scores, bins=bin_edges,
        alpha=0.65, color=COLOR_ALT, density=True,
        label=f"Top-25% future adverse outcome  (n={n_high:,})",
    )

    # ── Recommendation-band threshold markers ──────────────────────────────
    # 30 = Recommend→Caution boundary, 60 = Caution→Do-Not-Recommend boundary
    for x_thresh, label_txt, ls in (
        (30, "Caution (30)", "--"),
        (60, "Do Not Recommend (60)", "-."),
    ):
        ax.axvline(
            x=x_thresh, color="#555555", linewidth=1.6,
            linestyle=ls, label=label_txt,
        )

    ax.set_xlabel("Baseline Risk Score", fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.set_title(
        "Score Distribution: All vs. High Future Adverse Outcome",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "05_distribution_overlay.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_decile_boxplot(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 6 -- boxplot of future adverse outcome by score decile.

    Score deciles are defined by percentile boundaries so each box represents
    approximately 10% of the paired population.  The upward shift in box
    positions across deciles shows whether the risk ordering translates into a
    genuine distributional shift in future outcomes.
    """
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    decile_edges = np.percentile(scores, np.linspace(0, 100, 11))

    data_by_decile   = []
    labels_by_decile = []
    medians          = []
    ns               = []

    for i in range(10):
        lo   = decile_edges[i]
        hi   = decile_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == 9 \
               else (scores >= lo) & (scores < hi)
        vals = adverse[mask]
        data_by_decile.append(vals)
        medians.append(float(np.median(vals)) if len(vals) > 0 else 0.0)
        ns.append(int(mask.sum()))
        labels_by_decile.append(f"D{i + 1}")

    fig, ax = plt.subplots(figsize=(11, 6))

    bp = ax.boxplot(
        data_by_decile,
        positions=list(range(1, 11)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.35,
                        markerfacecolor=COLOR_MAIN, markeredgewidth=0),
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Colour boxes by decile: gradient from blue (low risk) to red (high risk)
    cmap = plt.get_cmap("RdYlBu_r")
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / 9.0))
        patch.set_alpha(0.75)

    # Annotate each box with n
    y_top = max(float(np.percentile(adverse, 95)) * 1.05, 1.0)
    for i, (med, n) in enumerate(zip(medians, ns)):
        ax.text(
            i + 1, y_top * 0.97,
            f"n={n:,}",
            ha="center", va="top",
            fontsize=FONT_ANNOT - 1, color="#333333",
        )

    ax.set_xticks(list(range(1, 11)))
    ax.set_xticklabels(labels_by_decile, fontsize=FONT_TICK)
    ax.set_xlabel("Score Decile  (D1 = lowest  ->  D10 = highest)",
                  fontsize=FONT_LABEL)
    ax.set_ylabel("Future Adverse Outcome Score", fontsize=FONT_LABEL)
    ax.set_title(
        "Distribution of Future Adverse Outcome by Score Decile",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.set_ylim(bottom=0, top=y_top)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, axis="y", linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "06_decile_boxplot.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ====================================================================== #
#  New plots (07 – 09)
# ====================================================================== #

def _plot_threshold_pr_f1(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 07 -- precision / recall / F1 vs. score threshold for two targets.

    Left panel: future adverse score >= 75th percentile of the paired population.
    Right panel: any future serious / willful / repeat violation.

    Business interpretation: the crossing point of precision and recall curves
    marks the threshold with the highest F1 — the optimal trade-off between
    flagging too many manufacturers (low precision) and missing future incidents
    (low recall) for a given target definition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    targets = [
        (
            rw_data.swr_75th_flags,
            "Target A: future adverse >= 75th pct",
            "score threshold",
        ),
        (
            rw_data.paired_swr_flags,
            "Target B: any future S/W/R event",
            "score threshold",
        ),
    ]

    for ax, (y_true, panel_title, _x_label) in zip(axes, targets):
        if len(y_true) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL)
            ax.set_title(panel_title, fontsize=FONT_LABEL, fontweight="bold")
            continue

        df = _compute_threshold_metrics(
            rw_data.paired_scores, y_true, EVAL_THRESHOLDS,
        )
        xs         = df["threshold"].values
        precision  = df["precision"].values
        recall     = df["recall"].values
        f1         = df["F1"].values
        prevalence = float(df["prevalence"].iloc[0])

        ax.plot(xs, precision, marker="o", linewidth=2.0,
                color=COLOR_MAIN,  label="Precision")
        ax.plot(xs, recall,    marker="s", linewidth=2.0,
                color=COLOR_ALT,   label="Recall")
        ax.plot(xs, f1,        marker="^", linewidth=2.0,
                color="#1a9641", linestyle="-.", label="F1")
        ax.axhline(y=prevalence, color=COLOR_REF, linewidth=1.2, linestyle=":",
                   label=f"Prevalence ({prevalence:.3f})")

        # Mark the threshold with the highest F1
        best_idx = int(np.argmax(f1))
        ax.annotate(
            f"Best F1\n@ {xs[best_idx]}",
            xy=(xs[best_idx], f1[best_idx]),
            xytext=(8, -20), textcoords="offset points",
            fontsize=FONT_ANNOT + 1,
            arrowprops=dict(arrowstyle="->", color="#222", lw=1.1),
        )

        ax.set_xlim(min(xs) - 5, max(xs) + 5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Score Threshold", fontsize=FONT_LABEL)
        ax.set_ylabel("Metric Value", fontsize=FONT_LABEL)
        ax.set_title(panel_title, fontsize=FONT_LABEL, fontweight="bold")
        ax.set_xticks(xs)
        ax.tick_params(labelsize=FONT_TICK)
        ax.legend(fontsize=FONT_LEGEND - 1)
        ax.grid(True, linewidth=0.5)

    fig.suptitle(
        "Threshold Precision / Recall / F1 by Binary Target",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(plots_dir, "07_threshold_pr_f1.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_recommendation_band_outcomes(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 08 -- grouped bar chart comparing future outcomes per recommendation band.

    Recommendation bands are defined by the same score cut-points used in
    RiskAssessor:
      Recommend          score <  30
      Proceed with Caution  30 <= score <= 60
      Do Not Recommend   score >  60

    Business interpretation: the bars directly validate that the recommendation
    categories align with realised future compliance outcomes.  If the "Do Not
    Recommend" band does not show materially worse future outcomes than
    "Recommend", the recommendation thresholds need recalibration.
    """
    scores   = rw_data.paired_scores
    outcomes = rw_data.paired_outcomes

    bands = [
        ("Recommend\n(< 30)",         scores < 30),
        ("Caution\n(30–60)",    (scores >= 30) & (scores <= 60)),
        ("Do Not Rec.\n(> 60)",        scores > 60),
    ]

    metrics = ["mean_adverse", "swr_rate", "fatality_rate"]
    metric_labels = [
        "Mean Adverse Score",
        "S/W/R Event Rate",
        "Fatality Rate",
    ]
    metric_colors = [COLOR_MAIN, COLOR_ALT, "#984ea3"]

    n_metrics = len(metrics)
    n_bands   = len(bands)
    x         = np.arange(n_bands)
    bar_w     = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    band_ns   = []
    band_vals = {m: [] for m in metrics}

    for _band_label, mask in bands:
        idx = np.where(mask)[0].tolist()
        band_ns.append(len(idx))
        adv  = [outcomes[i]["future_adverse_outcome_score"] for i in idx]
        swr  = [outcomes[i]["future_any_serious_or_willful_repeat"] for i in idx]
        fat  = [outcomes[i]["future_fatality_or_catastrophe"] for i in idx]
        band_vals["mean_adverse"].append(float(np.mean(adv)) if adv else 0.0)
        band_vals["swr_rate"].append(float(np.mean(swr)) if swr else 0.0)
        band_vals["fatality_rate"].append(float(np.mean(fat)) if fat else 0.0)

    # Normalise each metric to [0, 1] so all three fit on the same axis
    # while preserving relative differences within each metric.
    for metric_idx, (metric, label, color) in enumerate(
        zip(metrics, metric_labels, metric_colors)
    ):
        vals     = band_vals[metric]
        max_v    = max(vals) if max(vals) > 0 else 1.0
        norm_v   = [v / max_v for v in vals]
        offset   = (metric_idx - (n_metrics - 1) / 2) * bar_w
        bars_h   = ax.bar(
            x + offset, norm_v, bar_w,
            label=label, color=color, alpha=0.82, edgecolor="white",
        )
        # Annotate with the raw (un-normalised) value
        for bar, raw_v in zip(bars_h, vals):
            if raw_v == 0:
                continue
            fmt = f"{raw_v:.2f}" if metric == "mean_adverse" else f"{raw_v:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                fmt,
                ha="center", va="bottom",
                fontsize=FONT_ANNOT, color="#111111",
            )

    # Annotate n per band along the x-axis
    for i, n_band in enumerate(band_ns):
        ax.text(
            x[i], -0.08,
            f"n={n_band:,}",
            ha="center", va="top",
            fontsize=FONT_ANNOT, color="#333333",
            transform=ax.get_xaxis_transform(),
        )

    band_labels = [b[0] for b in bands]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=FONT_TICK)
    ax.set_ylabel("Normalised Metric Value  (raw value annotated)",
                  fontsize=FONT_LABEL)
    ax.set_title(
        "Future Outcomes by Recommendation Band",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.set_ylim(0, 1.30)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, axis="y", linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "08_rec_band_outcomes.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


def _plot_confidence_subgroup(
    rw_data: RealWorldData,
    plots_dir: str,
) -> str:
    """Plot 09 -- score vs. adverse outcome scatter per confidence tier.

    Three panels (one per confidence level: High / Medium / Low).  Each panel
    shows the raw scatter overlaid with a rolling-mean trend line.  The panel
    title includes n and Spearman rho so analysts can immediately see how much
    predictive signal remains at each evidence level.

    Business interpretation: a strong upward trend in the High-confidence
    panel (and a flat or noisy trend in the Low-confidence panel) confirms
    that evidence depth matters.  Buyers should treat Low-confidence scores
    as preliminary signals requiring additional due diligence.
    """
    from scipy.stats import spearmanr

    if not rw_data.confidence_tags:
        # Nothing to plot — return early with a blank placeholder
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Confidence tags not available",
                ha="center", va="center", transform=ax.transAxes)
        path = os.path.join(plots_dir, "09_confidence_subgroup.png")
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        return path

    tags    = rw_data.confidence_tags
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    # Group data per confidence tier
    groups = {"High": ([], []), "Medium": ([], []), "Low": ([], [])}
    for tag, sc, adv in zip(tags, scores, adverse):
        if tag in groups:
            groups[tag][0].append(float(sc))
            groups[tag][1].append(float(adv))

    tier_order  = ["High", "Medium", "Low"]
    tier_colors = ["#2166ac", "#f4a582", "#d6604d"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, tier, color in zip(axes, tier_order, tier_colors):
        sc_arr  = np.array(groups[tier][0])
        adv_arr = np.array(groups[tier][1])
        n       = len(sc_arr)

        if n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL)
            ax.set_title(f"{tier} confidence  (n=0)",
                         fontsize=FONT_LABEL, fontweight="bold")
            continue

        rho_str = "N/A"
        if n >= 3:
            rho_val = float(spearmanr(sc_arr, adv_arr)[0])
            rho_str = f"{rho_val:.3f}"

        # Scatter
        ax.scatter(sc_arr, adv_arr,
                   alpha=0.35, s=8, color=color, rasterized=True)

        # Rolling-mean smooth (only when n >= 10)
        if n >= 10:
            xs_sm, ys_sm = _rolling_smooth(
                sc_arr, adv_arr,
                window_frac=0.15, min_window=max(5, n // 10),
            )
            ax.plot(xs_sm, ys_sm, color="#222222", linewidth=2.2,
                    label="Rolling mean")

        ax.set_xlabel("Baseline Risk Score", fontsize=FONT_LABEL - 1)
        ax.set_title(
            f"{tier} confidence\n(n={n:,}  rho={rho_str})",
            fontsize=FONT_LABEL, fontweight="bold",
        )
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, linewidth=0.4)

    axes[0].set_ylabel("Future Adverse Outcome Score", fontsize=FONT_LABEL)

    fig.suptitle(
        "Score Predictive Value by Evidence-Depth (Confidence Tier)",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(plots_dir, "09_confidence_subgroup.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ====================================================================== #
#  Console report
# ====================================================================== #

def _print_cumulative_gains_table(
    rw_data: RealWorldData,
    top_ks: tuple = (0.05, 0.10, 0.20, 0.30),
) -> None:
    """Print a cumulative gains table for selected top-k% thresholds.

    For each threshold the table shows:
      - n                : establishments in the top-k%
      - Adverse Captured : % of total future adverse outcome in this group
      - Lift vs Random   : captured_pct / k  (1.0 = same as random)
    """
    scores        = rw_data.paired_scores
    adverse       = rw_data.paired_adverse_scores
    n             = len(scores)
    total_adverse = float(adverse.sum())
    if total_adverse == 0.0:
        total_adverse = 1.0

    order          = np.argsort(scores)[::-1]
    sorted_adverse = adverse[order]
    cumsum_adverse = np.cumsum(sorted_adverse)

    col_w = 66
    sep   = "-" * col_w
    print("\nCumulative Gains Table  (establishments sorted by baseline score, desc)")
    print(sep)
    print(f"  {'Top-K':>8}   {'n':>7}   {'Adverse Captured':>18}   {'Lift vs Random':>16}")
    print(sep)

    for k in top_ks:
        idx      = max(1, int(round(n * k)))
        captured = float(cumsum_adverse[idx - 1]) / total_adverse
        lift     = captured / k
        print(
            f"  Top {k * 100:>4.0f}%   {idx:>7,}   "
            f"{captured * 100:>17.1f}%   "
            f"{lift:>14.2f}x"
        )

    print(sep + "\n")


# ====================================================================== #
#  Public API
# ====================================================================== #

def generate_all_validation_plots(
    rw_data: RealWorldData,
    plots_dir: str = PLOTS_DIR,
) -> list:
    """Generate all 9 validation plots and save them to plots_dir.

    Plots 01-06 validate rank correlation, decile lift, and score distribution.
    Plots 07-09 validate threshold performance, recommendation bands, and
    confidence-tier predictive breakdown.

    Also prints the cumulative gains table to stdout.

    Args:
        rw_data   : populated RealWorldData instance (use RealWorldData.get())
        plots_dir : output directory (created if it does not exist)

    Returns:
        List of absolute paths to the saved PNG files.
    """
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nGenerating validation plots -> {plots_dir}")
    saved = []

    steps = [
        ("01  Scatter: score vs adverse (smooth)",         _plot_scatter_score_vs_adverse),
        ("02  Binned mean + 95% CI (quantile bins)",        _plot_binned_mean_outcome),
        ("03  Decile lift  (lift + n)",                     _plot_decile_lift),
        ("04  Top-K event capture curve",                   _plot_capture_curve),
        ("05  Score distribution + threshold markers",      _plot_distribution_overlay),
        ("06  Decile boxplot",                              _plot_decile_boxplot),
        ("07  Threshold precision/recall/F1  (2 targets)",  _plot_threshold_pr_f1),
        ("08  Recommendation-band outcome bars",            _plot_recommendation_band_outcomes),
        ("09  Confidence-tier scatter comparison",          _plot_confidence_subgroup),
    ]

    for label, plot_fn in steps:
        try:
            path = plot_fn(rw_data, plots_dir)
            saved.append(path)
            print(f"  [{label}]  saved -> {os.path.basename(path)}")
        except Exception as exc:  # pragma: no cover
            print(f"  [{label}]  ERROR: {exc}")

    # Console gains table
    _print_cumulative_gains_table(rw_data)

    print(f"Done. {len(saved)} plots written to {plots_dir}\n")
    return saved


# ====================================================================== #
#  Standalone entry point
# ====================================================================== #

if __name__ == "__main__":
    rw = RealWorldData.get()
    generate_all_validation_plots(rw)
