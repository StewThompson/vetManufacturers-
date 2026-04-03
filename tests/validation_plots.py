"""
tests/validation_plots.py

Generates diagnostic matplotlib plots for the multi-target risk model and
saves them to plots/ under the project root.

Plots produced
--------------
  01_composite_score_vs_adverse.png  -- Composite score vs future adverse
                                        outcome: scatter + rolling-mean trend
  02_binary_head_roc.png             -- ROC curves for both binary prediction
                                        heads (WR event, injury event)
  03_regression_head_actual_vs_pred.png -- Actual vs predicted for penalty
                                           and gravity regression heads
  04_score_decile_adverse.png        -- Mean future adverse outcome by score
                                        decile (monotonicity check)

Usage (standalone)
------------------
    python tests/validation_plots.py

Usage (from code)
-----------------
    from tests.validation_plots import generate_all_validation_plots
    from tests.validation.shared import RealWorldData
    generate_all_validation_plots(RealWorldData.get())
"""

import os
import sys
import math

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TESTS_DIR    = os.path.dirname(__file__)
for _p in (_PROJECT_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.validation.shared import RealWorldData, CACHE_DIR
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.scoring.multi_target_scorer import MultiTargetRiskScorer

# ── Output directory ──────────────────────────────────────────────────
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")

# ── Visual constants ──────────────────────────────────────────────────
FONT_TITLE  = 14
FONT_LABEL  = 12
FONT_TICK   = 10
FONT_ANNOT  = 9
FIG_SIZE    = (9, 6)
FIG_DPI     = 150
COLOR_WR    = "#2c7bb6"   # steel blue — WR head
COLOR_INJ   = "#d7191c"   # red — injury head
COLOR_PEN   = "#1a9641"   # green — penalty head
COLOR_GRAV  = "#d4a017"   # amber — gravity head
COLOR_REF   = "#aaaaaa"   # grey reference


# ====================================================================== #
#  Helper: get per-head predictions for the paired subset
# ====================================================================== #

def _get_paired_predictions(rw_data: RealWorldData):
    """Return (mt_scorer, preds) for the paired subset, or (None, [])."""
    mt_scorer = MultiTargetRiskScorer.load_if_exists(CACHE_DIR)
    if mt_scorer is None:
        return None, []

    # Build a boolean mask aligning hist_pop to future_outcomes
    paired_mask = np.array([o["has_future_data"] for o in rw_data.future_outcomes])
    if not paired_mask.any():
        return mt_scorer, []

    paired_X = rw_data.hist_X[paired_mask]
    paired_X_log = MLRiskScorer._log_transform_features(
        np.nan_to_num(paired_X, nan=0.0)
    )
    preds = mt_scorer.predict_batch(paired_X_log)
    return mt_scorer, preds


def _rolling_smooth(scores, outcomes, window_frac=0.08, min_window=50):
    sort_idx = np.argsort(scores)
    xs       = scores[sort_idx]
    ys       = outcomes[sort_idx]
    win      = max(min_window, int(len(xs) * window_frac))
    ys_sm = (
        pd.Series(ys)
        .rolling(window=win, center=True, min_periods=max(1, win // 4))
        .mean()
        .values
    )
    return xs, ys_sm


# ====================================================================== #
#  Plot 1: Composite score vs future adverse outcome
# ====================================================================== #

def _plot_composite_vs_adverse(rw_data: RealWorldData, plots_dir: str) -> str:
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.scatter(scores, adverse, alpha=0.3, s=6, color=COLOR_WR,
               rasterized=True, label="_nolegend_")

    xs_sm, ys_sm = _rolling_smooth(scores, adverse)
    ax.plot(xs_sm, ys_sm, color=COLOR_INJ, linewidth=2.5, label="Rolling mean (trend)")

    m, b   = np.polyfit(scores, adverse, 1)
    x_line = np.linspace(float(scores.min()), float(scores.max()), 300)
    ax.plot(x_line, m * x_line + b, color="#555", linewidth=1.2,
            linestyle="--", label=f"Linear fit (slope={m:.4f})", alpha=0.6)

    from scipy.stats import spearmanr
    rho = float(spearmanr(scores, adverse)[0])
    ax.set_title(f"Composite Score vs Future Adverse Outcome  (ρ={rho:+.3f})",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Composite Risk Score (0–100)", fontsize=FONT_LABEL)
    ax.set_ylabel("Future Adverse Outcome Score", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT + 1)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "01_composite_score_vs_adverse.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ====================================================================== #
#  Plot 2: ROC curves for both binary heads
# ====================================================================== #

def _plot_binary_head_roc(rw_data: RealWorldData, plots_dir: str,
                          preds: list) -> str:
    from sklearn.metrics import roc_curve, auc

    swr_flags     = rw_data.paired_swr_flags
    fatality_flags = rw_data.paired_fatality_flags

    if not preds:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "MT model not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_LABEL)
        path = os.path.join(plots_dir, "02_binary_head_roc.png")
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        return path

    p_wr  = np.array([p["p_serious_wr_event"] for p in preds])
    p_inj = np.array([p["p_injury_event"]      for p in preds])

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    heads = [
        (p_wr,  swr_flags,     COLOR_WR,  "p_serious_wr_event  (target: S/W/R flag)"),
        (p_inj, fatality_flags, COLOR_INJ, "p_injury_event      (target: fatality flag)"),
    ]

    for proba, y_true, color, label in heads:
        n_pos = int(y_true.sum())
        if n_pos < 5 or len(np.unique(y_true)) < 2:
            ax.plot([], [], color=color, label=f"{label}  [insufficient positives]")
            continue
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{label}\nAUROC={roc_auc:.3f}  n_pos={n_pos}")

    ax.plot([0, 1], [0, 1], color=COLOR_REF, linewidth=1.2, linestyle="--",
            label="Random baseline (AUROC=0.50)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate", fontsize=FONT_LABEL)
    ax.set_title("ROC Curves — Binary Prediction Heads", fontsize=FONT_TITLE,
                 fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT, loc="lower right")
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "02_binary_head_roc.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ====================================================================== #
#  Plot 3: Regression head actual vs predicted
# ====================================================================== #

def _plot_regression_heads(rw_data: RealWorldData, plots_dir: str,
                            preds: list) -> str:
    if not preds:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "MT model not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_LABEL)
        path = os.path.join(plots_dir, "03_regression_head_actual_vs_pred.png")
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        return path

    pred_penalty = np.array([p["expected_penalty_usd"] for p in preds])
    pred_gravity = np.array([p["gravity_score"]         for p in preds])

    actual_penalty = np.array([
        o["future_total_penalty"] for o in rw_data.paired_outcomes
    ])
    actual_gravity = np.array([
        o["future_adverse_outcome_score"] for o in rw_data.paired_outcomes
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        (pred_penalty, actual_penalty, COLOR_PEN,
         "Penalty Head", "Predicted ($)", "Actual Future Penalty ($)"),
        (pred_gravity, actual_gravity, COLOR_GRAV,
         "Gravity Head", "Predicted Gravity Score", "Actual Adverse Outcome Score"),
    ]

    for ax, (pred, actual, color, title, xlabel, ylabel) in zip(axes, panels):
        # Clip extreme outliers for display (keep 99th pct)
        p99_pred   = float(np.percentile(pred,   99))
        p99_actual = float(np.percentile(actual, 99))
        mask = (pred <= p99_pred) & (actual <= p99_actual)

        ax.scatter(pred[mask], actual[mask], alpha=0.3, s=5,
                   color=color, rasterized=True)

        # Identity line
        xy_max = max(float(pred[mask].max()), float(actual[mask].max()))
        ax.plot([0, xy_max], [0, xy_max], color=COLOR_REF,
                linewidth=1.5, linestyle="--", label="Perfect prediction")

        from scipy.stats import spearmanr, pearsonr
        rho = float(spearmanr(pred, actual)[0])
        r   = float(pearsonr(pred, actual)[0])

        ax.set_title(f"{title}  (ρ={rho:+.3f}  r={r:+.3f})",
                     fontsize=FONT_LABEL, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL - 1)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL - 1)
        ax.tick_params(labelsize=FONT_TICK)
        ax.legend(fontsize=FONT_ANNOT)
        ax.grid(True, linewidth=0.4)

    fig.suptitle("Regression Heads: Actual vs Predicted",
                 fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "03_regression_head_actual_vs_pred.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ====================================================================== #
#  Plot 4: Score decile → mean future adverse (monotonicity check)
# ====================================================================== #

def _plot_score_decile_adverse(rw_data: RealWorldData, plots_dir: str) -> str:
    scores  = rw_data.paired_scores
    adverse = rw_data.paired_adverse_scores
    n       = len(scores)

    decile_edges = np.percentile(scores, np.linspace(0, 100, 11))
    decile_means = []
    decile_ns    = []
    decile_labels = []

    for i in range(10):
        lo   = decile_edges[i]
        hi   = decile_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == 9 \
               else (scores >= lo) & (scores < hi)
        vals = adverse[mask]
        decile_means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
        decile_ns.append(int(mask.sum()))
        decile_labels.append(f"D{i+1}")

    pop_mean = float(adverse.mean())
    lifts    = [m / max(pop_mean, 1e-9) for m in decile_means]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bar_colors = [COLOR_WR if lift >= 1.0 else COLOR_INJ for lift in lifts]
    bars = ax.bar(range(1, 11), decile_means, color=bar_colors,
                  edgecolor="white", width=0.7)

    ax.axhline(y=pop_mean, color=COLOR_REF, linewidth=1.8, linestyle="--",
               label=f"Population mean ({pop_mean:.2f})")

    y_max = max(decile_means) if decile_means else 1.0
    for i, (bar, mean, lift, n_d) in enumerate(
        zip(bars, decile_means, lifts, decile_ns)
    ):
        bx = bar.get_x() + bar.get_width() / 2
        bh = bar.get_height()
        ax.text(bx, bh + y_max * 0.015, f"{lift:.2f}×",
                ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold")
        if bh > y_max * 0.08:
            ax.text(bx, bh * 0.4, f"n={n_d:,}",
                    ha="center", va="center", fontsize=FONT_ANNOT - 1,
                    color="white", fontweight="bold")

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(decile_labels, fontsize=FONT_TICK)
    ax.set_xlabel("Score Decile  (D1 = lowest  →  D10 = highest)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean Future Adverse Outcome Score", fontsize=FONT_LABEL)
    ax.set_title("Score Decile → Mean Future Adverse Outcome  (lift annotated)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(0, y_max * 1.18)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT + 1)
    ax.grid(True, axis="y", linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(plots_dir, "04_score_decile_adverse.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ====================================================================== #
#  Public API
# ====================================================================== #

def generate_all_validation_plots(
    rw_data: RealWorldData,
    plots_dir: str = PLOTS_DIR,
) -> list:
    """Generate the 4 multi-target model assessment plots.

    Plot 1: Composite score vs future adverse outcome (rank correlation check)
    Plot 2: ROC curves for binary heads (WR event, injury event)
    Plot 3: Actual vs predicted for regression heads (penalty, gravity)
    Plot 4: Score decile → mean future adverse outcome (monotonicity)

    Args:
        rw_data   : populated RealWorldData instance (use RealWorldData.get())
        plots_dir : output directory (created if it does not exist)

    Returns:
        List of absolute paths to the saved PNG files.
    """
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nGenerating multi-target validation plots -> {plots_dir}")

    mt_scorer, preds = _get_paired_predictions(rw_data)
    if mt_scorer is None:
        print("  WARNING: MT model not found in cache — plots 2 and 3 will be blank.")

    steps = [
        ("01  Composite score vs future adverse",       lambda rd, pd: _plot_composite_vs_adverse(rd, pd)),
        ("02  Binary head ROC curves",                  lambda rd, pd: _plot_binary_head_roc(rd, pd, preds)),
        ("03  Regression head actual vs predicted",     lambda rd, pd: _plot_regression_heads(rd, pd, preds)),
        ("04  Score decile → mean adverse (decile lift)", lambda rd, pd: _plot_score_decile_adverse(rd, pd)),
    ]

    saved = []
    for label, plot_fn in steps:
        try:
            path = plot_fn(rw_data, plots_dir)
            saved.append(path)
            print(f"  [{label}]  saved -> {os.path.basename(path)}")
        except Exception as exc:
            print(f"  [{label}]  ERROR: {exc}")

    print(f"Done. {len(saved)}/4 plots written to {plots_dir}\n")
    return saved


# ====================================================================== #
#  Standalone entry point
# ====================================================================== #

if __name__ == "__main__":
    rw = RealWorldData.get()
    generate_all_validation_plots(rw)
