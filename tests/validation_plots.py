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

import math
import os
import sys

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
    # Use the actual gravity-weighted-score target (Σ gravity × viol_weight).
    # NOTE: earlier versions incorrectly used future_adverse_outcome_score here,
    # causing vertical-stripe artifacts and misleading ρ/r values because the
    # gravity head was being compared against a different composite target.
    actual_gravity = np.array([
        o.get("future_gravity_weighted_score", 0.0)
        for o in rw_data.paired_outcomes
    ])

    from scipy.stats import spearmanr, pearsonr

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: Penalty — log1p scale on both axes ───────────────────────
    # Monetary predictions span 3+ orders of magnitude ($0–$130k+).
    # Log-log scale is the standard display for regression across wide ranges;
    # it reveals rank correlation far better than linear scale where
    # near-zero cases dominate the visual space.
    ax = axes[0]
    log_pred   = np.log1p(pred_penalty)
    log_actual = np.log1p(actual_penalty)

    # Clip both at 99th pct for display (remove extreme outliers)
    p99_lp = float(np.percentile(log_pred,   99))
    p99_la = float(np.percentile(log_actual, 99))
    mask_p = (log_pred <= p99_lp) & (log_actual <= p99_la)

    ax.scatter(log_pred[mask_p], log_actual[mask_p], alpha=0.25, s=5,
               color=COLOR_PEN, rasterized=True)

    xy_max_p = max(float(log_pred[mask_p].max()), float(log_actual[mask_p].max()))
    ax.plot([0, xy_max_p], [0, xy_max_p], color=COLOR_REF,
            linewidth=1.5, linestyle="--", label="Perfect prediction")

    rho_p = float(spearmanr(pred_penalty, actual_penalty)[0])
    r_p   = float(pearsonr(log_pred, log_actual)[0])

    ax.set_title(f"Penalty Head  (ρ={rho_p:+.3f}  r(log)={r_p:+.3f})",
                 fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xlabel("log1p(Predicted $)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("log1p(Actual Future Penalty $)", fontsize=FONT_LABEL - 1)

    # Add readable tick labels in dollar scale
    tick_vals = [0, 100, 1_000, 5_000, 20_000, 100_000]
    tick_pos  = [math.log1p(v) for v in tick_vals]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"${v:,.0f}" if v > 0 else "$0" for v in tick_vals],
                       fontsize=FONT_TICK - 1, rotation=30, ha="right")
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"${v:,.0f}" if v > 0 else "$0" for v in tick_vals],
                       fontsize=FONT_TICK - 1)

    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT)
    ax.grid(True, linewidth=0.4)

    # ── Panel B: Gravity — log1p scale ────────────────────────────────────
    # gravity_weighted_score = Σ(gravity × viol_weight) spans 0–1500+.
    # Log1p scale compresses the tail for better visual clarity.
    ax = axes[1]
    log_pred_g   = np.log1p(np.maximum(0.0, pred_gravity))
    log_actual_g = np.log1p(np.maximum(0.0, actual_gravity))

    p99_lpg = float(np.percentile(log_pred_g,   99))
    p99_lag = float(np.percentile(log_actual_g, 99))
    mask_g = (log_pred_g <= p99_lpg) & (log_actual_g <= p99_lag)

    ax.scatter(log_pred_g[mask_g], log_actual_g[mask_g], alpha=0.25, s=5,
               color=COLOR_GRAV, rasterized=True)

    xy_max_g = max(float(log_pred_g[mask_g].max()), float(log_actual_g[mask_g].max()))
    ax.plot([0, xy_max_g], [0, xy_max_g], color=COLOR_REF,
            linewidth=1.5, linestyle="--", label="Perfect prediction")

    rho_g = float(spearmanr(pred_gravity, actual_gravity)[0])
    r_g   = float(pearsonr(log_pred_g, log_actual_g)[0])

    ax.set_title(f"Gravity Head  (ρ={rho_g:+.3f}  r(log)={r_g:+.3f})",
                 fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xlabel("log1p(Predicted Gravity Score)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("log1p(Actual Gravity-Weighted Score)", fontsize=FONT_LABEL - 1)

    grav_tick_vals = [0, 1, 5, 20, 50, 150, 500]
    grav_tick_pos  = [math.log1p(v) for v in grav_tick_vals]
    ax.set_xticks(grav_tick_pos)
    ax.set_xticklabels([str(v) for v in grav_tick_vals],
                       fontsize=FONT_TICK - 1)
    ax.set_yticks(grav_tick_pos)
    ax.set_yticklabels([str(v) for v in grav_tick_vals],
                       fontsize=FONT_TICK - 1)

    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT)
    ax.grid(True, linewidth=0.4)

    fig.suptitle("Regression Heads: Actual vs Predicted  (log1p scale)",
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
#  Plot 5: Stage 1 — Inspection Exposure (ROC + Calibration)
# ====================================================================== #

def _plot_stage1_inspection(rw_data: RealWorldData, plots_dir: str,
                            preds: list) -> str:
    """Stage 1 evaluation: ROC-AUC, PR-AUC, and calibration curve for
    the inspection exposure model (CONDITIONAL label: was this establishment
    inspected in the future window?)."""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    if not preds or "pred_p_inspection" not in preds[0]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "Stage 1 predictions not available", ha="center",
                va="center", transform=ax.transAxes, fontsize=FONT_LABEL)
        path = os.path.join(plots_dir, "05_stage1_inspection_exposure.png")
        fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
        return path

    # For the paired subset, future_has_inspection is always 1 — so we need
    # to check if we have a mix of inspected/not-inspected in the full data.
    # Since validation_plots only has the paired subset, we plot ROC vs
    # the SWR flag to show that inspection probability discriminates risk.
    p_insp = np.array([p["pred_p_inspection"] for p in preds])
    swr    = rw_data.paired_swr_flags

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: ROC curve (inspection prob vs SWR outcome)
    ax = axes[0]
    n_pos = int(swr.sum())
    if n_pos >= 5 and len(np.unique(swr)) >= 2:
        fpr, tpr, _ = roc_curve(swr, p_insp)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLOR_WR, linewidth=2.2,
                label=f"Inspection exposure\nAUROC={roc_auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color=COLOR_REF, linewidth=1.2, linestyle="--",
            label="Random")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR", fontsize=FONT_LABEL)
    ax.set_ylabel("TPR", fontsize=FONT_LABEL)
    ax.set_title("Stage 1: Inspection Exposure ROC\n(unconditional)",
                 fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_ANNOT, loc="lower right")
    ax.grid(True, linewidth=0.5)

    # Panel B: Calibration-style decile plot
    ax = axes[1]
    n = len(p_insp)
    decile_labels = pd.qcut(pd.Series(p_insp).rank(method="first"),
                            q=10, labels=False, duplicates="drop")
    decile_means = []
    for d in sorted(decile_labels.unique()):
        mask = decile_labels == d
        decile_means.append(float(p_insp[mask.values].mean()))
    ax.bar(range(1, len(decile_means) + 1), decile_means, color=COLOR_WR,
           edgecolor="white", width=0.7)
    ax.set_xlabel("Predicted Inspection Prob Decile", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean P(inspection)", fontsize=FONT_LABEL)
    ax.set_title("Stage 1: Inspection Probability Deciles",
                 fontsize=FONT_LABEL, fontweight="bold")
    ax.grid(True, axis="y", linewidth=0.5)

    fig.suptitle("Stage 1 — Inspection Exposure Model", fontsize=FONT_TITLE,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "05_stage1_inspection_exposure.png")
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    return path


# ====================================================================== #
#  Plot 6: Stage 2 — Violation | Inspection (conditional ROC)
# ====================================================================== #

def _plot_stage2_violation(rw_data: RealWorldData, plots_dir: str,
                           preds: list) -> str:
    """Stage 2 evaluation: ROC-AUC for violation|inspection and
    serious|inspection models (CONDITIONAL — only on inspected rows)."""
    from sklearn.metrics import roc_curve, auc

    if not preds or "pred_p_violation_given_insp" not in preds[0]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "Stage 2 predictions not available", ha="center",
                va="center", transform=ax.transAxes, fontsize=FONT_LABEL)
        path = os.path.join(plots_dir, "06_stage2_violation_given_insp.png")
        fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
        return path

    p_viol = np.array([p["pred_p_violation_given_insp"] for p in preds])
    p_ser  = np.array([p["pred_p_serious_given_insp"]  for p in preds])

    # Actual targets for the paired subset (all inspected by definition)
    n_viols = np.array([o.get("future_n_violations", 0) or 0
                        for o in rw_data.paired_outcomes])
    y_has_viol = (n_viols > 0).astype(int)
    y_swr = rw_data.paired_swr_flags

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Violation | Inspection ROC
    ax = axes[0]
    n_pos = int(y_has_viol.sum())
    if n_pos >= 5 and len(np.unique(y_has_viol)) >= 2:
        fpr, tpr, _ = roc_curve(y_has_viol, p_viol)
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLOR_PEN, linewidth=2.2,
                label=f"P(violation|insp)\nAUROC={roc_val:.3f}  n_pos={n_pos}")
    ax.plot([0, 1], [0, 1], color=COLOR_REF, linewidth=1.2, linestyle="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR", fontsize=FONT_LABEL)
    ax.set_ylabel("TPR", fontsize=FONT_LABEL)
    ax.set_title("Any Violation | Inspection\n(conditional Stage 2)",
                 fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_ANNOT, loc="lower right")
    ax.grid(True, linewidth=0.5)

    # Panel B: Serious | Inspection ROC
    ax = axes[1]
    n_pos_s = int(y_swr.sum())
    if n_pos_s >= 5 and len(np.unique(y_swr)) >= 2:
        fpr, tpr, _ = roc_curve(y_swr, p_ser)
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLOR_INJ, linewidth=2.2,
                label=f"P(serious|insp)\nAUROC={roc_val:.3f}  n_pos={n_pos_s}")
    ax.plot([0, 1], [0, 1], color=COLOR_REF, linewidth=1.2, linestyle="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR", fontsize=FONT_LABEL)
    ax.set_ylabel("TPR", fontsize=FONT_LABEL)
    ax.set_title("Serious/W/R | Inspection\n(conditional Stage 2)",
                 fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_ANNOT, loc="lower right")
    ax.grid(True, linewidth=0.5)

    fig.suptitle("Stage 2 — Violation given Inspection (conditional ROC)",
                 fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "06_stage2_violation_given_insp.png")
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    return path


# ====================================================================== #
#  Plot 7: Stage 3 — Magnitude | Violation (conditional scatter)
# ====================================================================== #

def _plot_stage3_magnitude(rw_data: RealWorldData, plots_dir: str,
                           preds: list) -> str:
    """Stage 3 evaluation: conditional scatter plots for penalty and gravity
    regressors.  Only shown for establishments with ≥1 future violation
    (the training population for Stage 3)."""
    from scipy.stats import spearmanr, pearsonr

    if not preds or "pred_penalty_given_viol" not in preds[0]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "Stage 3 predictions not available", ha="center",
                va="center", transform=ax.transAxes, fontsize=FONT_LABEL)
        path = os.path.join(plots_dir, "07_stage3_magnitude_given_viol.png")
        fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
        return path

    # Filter to violation-only rows (Stage 3 training population)
    n_viols = np.array([o.get("future_n_violations", 0) or 0
                        for o in rw_data.paired_outcomes])
    viol_mask = n_viols > 0

    if viol_mask.sum() < 20:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"Only {viol_mask.sum()} violation rows (need ≥20)",
                ha="center", va="center", transform=ax.transAxes)
        path = os.path.join(plots_dir, "07_stage3_magnitude_given_viol.png")
        fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
        return path

    # Conditional predictions (what the model predicts GIVEN a violation)
    cond_pen  = np.array([p["pred_penalty_given_viol"] for p in preds])[viol_mask]
    cond_grav = np.array([p["pred_gravity_given_viol"] for p in preds])[viol_mask]

    actual_pen  = np.array([o["future_total_penalty"] for o in rw_data.paired_outcomes])[viol_mask]
    actual_grav = np.array([o.get("future_gravity_weighted_score", 0.0)
                            for o in rw_data.paired_outcomes])[viol_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Conditional Penalty (log1p scale)
    ax = axes[0]
    log_cp = np.log1p(cond_pen)
    log_ap = np.log1p(actual_pen)
    ax.scatter(log_cp, log_ap, alpha=0.25, s=5, color=COLOR_PEN, rasterized=True)
    xy_max = max(float(log_cp.max()), float(log_ap.max()))
    ax.plot([0, xy_max], [0, xy_max], color=COLOR_REF, linewidth=1.5,
            linestyle="--", label="Perfect prediction")
    rho_p = float(spearmanr(cond_pen, actual_pen)[0])
    r_p   = float(pearsonr(log_cp, log_ap)[0])
    ax.set_title(f"Penalty|Viol  (ρ={rho_p:+.3f}  r(log)={r_p:+.3f})\n"
                 f"[conditional: {int(viol_mask.sum())} violation rows only]",
                 fontsize=FONT_LABEL - 1, fontweight="bold")
    ax.set_xlabel("log1p(Predicted $)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("log1p(Actual $)", fontsize=FONT_LABEL - 1)
    ax.legend(fontsize=FONT_ANNOT); ax.grid(True, linewidth=0.4)

    # Panel B: Conditional Gravity (log1p scale)
    ax = axes[1]
    log_cg = np.log1p(np.maximum(0.0, cond_grav))
    log_ag = np.log1p(np.maximum(0.0, actual_grav))
    ax.scatter(log_cg, log_ag, alpha=0.25, s=5, color=COLOR_GRAV, rasterized=True)
    xy_max_g = max(float(log_cg.max()), float(log_ag.max()))
    ax.plot([0, xy_max_g], [0, xy_max_g], color=COLOR_REF, linewidth=1.5,
            linestyle="--", label="Perfect prediction")
    rho_g = float(spearmanr(cond_grav, actual_grav)[0])
    r_g   = float(pearsonr(log_cg, log_ag)[0])
    ax.set_title(f"Gravity|Viol  (ρ={rho_g:+.3f}  r(log)={r_g:+.3f})\n"
                 f"[conditional: {int(viol_mask.sum())} violation rows only]",
                 fontsize=FONT_LABEL - 1, fontweight="bold")
    ax.set_xlabel("log1p(Predicted Gravity)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("log1p(Actual Gravity)", fontsize=FONT_LABEL - 1)
    ax.legend(fontsize=FONT_ANNOT); ax.grid(True, linewidth=0.4)

    fig.suptitle("Stage 3 — Magnitude given Violation (conditional scatter)",
                 fontsize=FONT_TITLE, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "07_stage3_magnitude_given_viol.png")
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    return path


# ====================================================================== #
#  Plot 8: Unconditional composite — expected values vs realized outcome
# ====================================================================== #

def _plot_unconditional_decile_table(rw_data: RealWorldData, plots_dir: str,
                                     preds: list) -> str:
    """Unconditional business-facing output: expected penalty/gravity composite
    vs realized future adverse outcome by predicted risk decile."""
    if not preds or "expected_penalty" not in preds[0]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "Unconditional predictions not available", ha="center",
                va="center", transform=ax.transAxes, fontsize=FONT_LABEL)
        path = os.path.join(plots_dir, "08_unconditional_risk_decile.png")
        fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
        return path

    from scipy.stats import spearmanr

    # Composite expected score
    exp_penalty = np.array([p["expected_penalty"] for p in preds])
    adverse     = rw_data.paired_adverse_scores

    # Normalize to [0, 1] for composite
    pen_norm = exp_penalty / (exp_penalty + 75_000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Expected penalty vs adverse scatter
    ax = axes[0]
    ax.scatter(np.log1p(exp_penalty), adverse, alpha=0.25, s=5,
               color=COLOR_PEN, rasterized=True)
    rho = float(spearmanr(exp_penalty, adverse)[0])
    ax.set_title(f"Unconditional E[penalty] vs Adverse  (ρ={rho:+.3f})",
                 fontsize=FONT_LABEL - 1, fontweight="bold")
    ax.set_xlabel("log1p(Expected Penalty $)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("Future Adverse Outcome Score", fontsize=FONT_LABEL - 1)
    ax.grid(True, linewidth=0.4)

    # Panel B: Decile table — mean realized adverse by predicted risk decile
    ax = axes[1]
    scores = rw_data.paired_scores
    decile_edges = np.percentile(scores, np.linspace(0, 100, 11))
    decile_means = []
    for i in range(10):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == 9 else (scores >= lo) & (scores < hi)
        vals = adverse[mask]
        decile_means.append(float(vals.mean()) if len(vals) > 0 else 0.0)

    pop_mean = float(adverse.mean())
    lifts = [m / max(pop_mean, 1e-9) for m in decile_means]
    bar_colors = [COLOR_WR if l >= 1.0 else "#cccccc" for l in lifts]
    bars = ax.bar(range(1, 11), decile_means, color=bar_colors,
                  edgecolor="white", width=0.7)
    ax.axhline(y=pop_mean, color=COLOR_REF, linewidth=1.8, linestyle="--",
               label=f"Pop mean ({pop_mean:.2f})")

    for i, (bar, lift) in enumerate(zip(bars, lifts)):
        bx = bar.get_x() + bar.get_width() / 2
        bh = bar.get_height()
        ax.text(bx, bh + max(decile_means) * 0.015, f"{lift:.2f}×",
                ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold")

    ax.set_xlabel("Composite Risk Score Decile (unconditional)", fontsize=FONT_LABEL - 1)
    ax.set_ylabel("Mean Realized Adverse Outcome", fontsize=FONT_LABEL - 1)
    ax.set_title("Unconditional Risk Decile → Realized Outcome",
                 fontsize=FONT_LABEL - 1, fontweight="bold")
    ax.legend(fontsize=FONT_ANNOT); ax.grid(True, axis="y", linewidth=0.5)

    fig.suptitle("Unconditional Business-Facing Outputs", fontsize=FONT_TITLE,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(plots_dir, "08_unconditional_risk_decile.png")
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    return path


def generate_all_validation_plots(
    rw_data: RealWorldData,
    plots_dir: str = PLOTS_DIR,
) -> list:
    """Generate all multi-target model assessment plots.

    Plots 01-04: Original diagnostics (backward compat)
    Plots 05-08: Sequential conditional pipeline stage evaluation

    Plot 1: Composite score vs future adverse outcome (rank correlation check)
    Plot 2: ROC curves for binary heads (WR event, injury event)
    Plot 3: Actual vs predicted for regression heads (penalty, gravity)
    Plot 4: Score decile → mean future adverse outcome (monotonicity)
    Plot 5: Stage 1 — Inspection exposure (ROC + calibration)
    Plot 6: Stage 2 — Violation|Inspection (conditional ROC)
    Plot 7: Stage 3 — Magnitude|Violation (conditional scatter)
    Plot 8: Unconditional composite — risk decile vs realized outcome
    """
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nGenerating multi-target validation plots -> {plots_dir}")

    mt_scorer, preds = _get_paired_predictions(rw_data)
    if mt_scorer is None:
        print("  WARNING: MT model not found in cache — plots 2-8 will be blank.")

    steps = [
        ("01  Composite score vs future adverse",        lambda rd, pd: _plot_composite_vs_adverse(rd, pd)),
        ("02  Binary head ROC curves",                   lambda rd, pd: _plot_binary_head_roc(rd, pd, preds)),
        ("03  Regression head actual vs predicted",      lambda rd, pd: _plot_regression_heads(rd, pd, preds)),
        ("04  Score decile → mean adverse (decile lift)", lambda rd, pd: _plot_score_decile_adverse(rd, pd)),
        ("05  Stage 1: Inspection exposure",             lambda rd, pd: _plot_stage1_inspection(rd, pd, preds)),
        ("06  Stage 2: Violation given inspection",      lambda rd, pd: _plot_stage2_violation(rd, pd, preds)),
        ("07  Stage 3: Magnitude given violation",       lambda rd, pd: _plot_stage3_magnitude(rd, pd, preds)),
        ("08  Unconditional risk decile",                lambda rd, pd: _plot_unconditional_decile_table(rd, pd, preds)),
    ]

    saved = []
    for label, plot_fn in steps:
        try:
            path = plot_fn(rw_data, plots_dir)
            saved.append(path)
            print(f"  [{label}]  saved -> {os.path.basename(path)}")
        except Exception as exc:
            print(f"  [{label}]  ERROR: {exc}")

    print(f"Done. {len(saved)}/{len(steps)} plots written to {plots_dir}\n")
    return saved


# ====================================================================== #
#  Standalone entry point
# ====================================================================== #

if __name__ == "__main__":
    rw = RealWorldData.get()
    generate_all_validation_plots(rw)
