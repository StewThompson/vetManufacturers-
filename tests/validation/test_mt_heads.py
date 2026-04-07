"""tests/validation/test_mt_heads.py
Tests 1-5: data integrity, WR/serious AUROC, large-penalty AUROC,
log-penalty Spearman, citation Spearman.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import math
import numpy as np
import pytest
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from collections import defaultdict

from tests.validation.mt_shared import (
    CUTOFF_DATE, CACHE_DIR, MIN_PAIRED, MIN_BINARY_POS, MIN_INDUSTRY_N,
    SPEARMAN_MINIMUM, SPEARMAN_STRONG, SPEARMAN_HIGH,
    AUROC_MINIMUM, AUROC_STRONG, AUROC_HIGH,
    AUROC_P_EVENT_TARGET, AUROC_P_INJURY_TARGET,
    LIFT_MINIMUM, LIFT_STRONG, LIFT_HIGH,
    CAPTURE_MINIMUM, CAPTURE_STRONG,
    REGRESSION_SPEARMAN, MAX_CALIBRATION_VIOLATIONS,
    BRIER_SS_MIN, ECE_MAX,
    BRIER_SS_MIN_P_INJURY, ECE_MAX_P_INJURY,
    CALIB_SLOPE_MIN_P_INJURY, CALIB_SLOPE_MAX_P_INJURY,
    CALIB_SLOPE_MIN, CALIB_SLOPE_MAX, CALIB_INTERCEPT_MIN, CALIB_INTERCEPT_MAX,
    PR_AUC_RATIO_P_EVENT, PR_AUC_RATIO_P_INJURY, PR_AUC_AP_FLOOR_EVENT,
    _get_scorer, _get_mt_scorer, _get_val_data,
    _read_csv, _load_validation_data,
)
import warnings
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.linear_model import LinearRegression
from src.scoring.penalty_percentiles import (
    load_percentiles,
    CACHE_FILENAME as PERCENTILE_CACHE,
)

def test_validation_data_volume():
    """Check that enough paired establishments exist at the validation cutoff."""
    X, rows, _ = _get_val_data()
    pytest.skip("Skipped: no data") if not rows else None
    n = len(rows)
    assert n >= MIN_PAIRED, (
        f"Only {n} paired establishments at cutoff {CUTOFF_DATE}; "
        f"need >= {MIN_PAIRED} for meaningful validation."
    )
    print(f"\n  Validation sample: {n:,} rows at cutoff={CUTOFF_DATE}")


def test_feature_shape():
    """Feature vectors must have the expected shape (47 features)."""
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")
    scorer = _get_scorer()
    expected = len(scorer.FEATURE_NAMES)
    assert X.shape[1] == expected, (
        f"Feature shape mismatch: got {X.shape[1]}, expected {expected}"
    )


def test_target_distributions():
    """Target distributions must have a minimum positive rate for binary heads."""
    _, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    wr_rate  = np.mean([r["any_wr_serious"]  for r in rows])
    inj_rate = np.mean([r["any_injury_fatal"] for r in rows])
    print(f"\n  WR/Serious positive rate:    {wr_rate:.1%}")
    print(f"  Hospitalization/Fatal rate:  {inj_rate:.1%}")

    # OSHA enforcement — at least 5% positive rate required for WR/Serious
    assert wr_rate >= 0.05, f"WR/serious positive rate too low: {wr_rate:.1%}"
    assert wr_rate <= 0.95, f"WR/serious positive rate too high: {wr_rate:.1%}"
    # Injury/fatal head may be sparse — just ensure it's not empty
    assert inj_rate > 0.0, f"Injury/fatal rate is exactly 0 — label generation may be broken"


def test_pos_label_rate_not_suppressed():
    """Guard against COVID-like inspection suppression silently degrading labels.

    During the 2020-2021 COVID period OSHA inspection volume collapsed, causing
    the any_wr_serious positive rate to drop well below 35%.  The 2022 training
    cutoff should keep the rate above this floor.  If it drops below 35% again,
    the training-cutoff or outcome-window configuration needs review.
    """
    _, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    wr_rate = np.mean([r["any_wr_serious"] for r in rows])
    print(f"\n  WR/Serious positive rate (suppression guard): {wr_rate:.1%}")
    assert wr_rate >= 0.35, (
        f"any_wr_serious positive rate {wr_rate:.1%} is below the 35% floor. "
        f"This may indicate COVID-era inspection suppression in the label window. "
        f"Check TEMPORAL_LABEL_CUTOFF and outcome_end date."
    )


def test_penalty_percentile_thresholds_exist():
    """Penalty percentile file must exist and have a global fallback entry."""
    thresh_path = os.path.join(CACHE_DIR, PERCENTILE_CACHE)
    if not os.path.exists(thresh_path):
        pytest.skip("penalty_percentiles.json not found; run build_cache.py first")
    thresholds = load_percentiles(thresh_path)
    assert "__global__" in thresholds, "Missing __global__ fallback in penalty thresholds"
    g = thresholds["__global__"]
    assert g["p75"] < g["p90"] < g["p95"], (
        f"Global thresholds not monotone: P75={g['p75']:.0f} P90={g['p90']:.0f} P95={g['p95']:.0f}"
    )
    print(
        f"\n  Global thresholds: P75=${g['p75']:,.0f}  "
        f"P90=${g['p90']:,.0f}  P95=${g['p95']:,.0f}"
    )


def test_model_loaded():
    """Multi-target model must be loadable from cache."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip(
            "multi_target_model.pkl not found; run 'python scripts/build_cache.py' first"
        )
    assert mt.is_fitted, "MultiTargetRiskScorer loaded but reports is_fitted=False"


# ====================================================================== #
#  Test 2: Binary head — WR/Serious AUROC
# ====================================================================== #

def test_wr_serious_auroc_minimum():
    """AUROC for WR/Serious binary head must be >= AUROC_P_EVENT_TARGET on held-out data."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    n_pos = y_true.sum()
    if n_pos < MIN_BINARY_POS:
        pytest.skip(f"Only {n_pos} positive WR/Serious examples; need >= {MIN_BINARY_POS}")

    preds = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])

    auroc = roc_auc_score(y_true, y_pred)
    print(f"\n  WR/Serious AUROC: {auroc:.4f} (target >= {AUROC_P_EVENT_TARGET})")
    assert auroc >= AUROC_P_EVENT_TARGET, (
        f"WR/Serious AUROC {auroc:.4f} below target {AUROC_P_EVENT_TARGET}"
    )


def test_wr_serious_auroc_strong():
    """AUROC >= AUROC_STRONG — marks 'strong model' quality level."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    n_pos = y_true.sum()
    if n_pos < MIN_BINARY_POS:
        pytest.skip(f"Insufficient positives ({n_pos})")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])
    auroc  = roc_auc_score(y_true, y_pred)
    print(f"\n  WR/Serious AUROC: {auroc:.4f} (strong target: {AUROC_STRONG})")
    assert auroc >= AUROC_STRONG, (
        f"WR/Serious AUROC {auroc:.4f} below strong target {AUROC_STRONG}"
    )


def test_wr_serious_precision_at_k():
    """Precision@10% must be > base rate for WR/Serious head."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])

    k       = max(1, int(len(y_true) * 0.10))
    top_k   = np.argsort(y_pred)[::-1][:k]
    prec_k  = y_true[top_k].mean()
    base    = y_true.mean()

    print(f"\n  WR/Serious Precision@10%: {prec_k:.3f} vs base rate {base:.3f}")
    assert prec_k > base, (
        f"Precision@10% ({prec_k:.3f}) not better than base rate ({base:.3f})"
    )


# ====================================================================== #
#  Test 3: Binary head — Hospitalization/Fatality (p_injury) AUROC
# ====================================================================== #

def test_p_injury_auroc_minimum():
    """AUROC for hospitalization/fatality head must be >= AUROC_P_INJURY_TARGET."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_injury_fatal"] for r in rows], dtype=int)
    n_pos  = y_true.sum()
    if n_pos < MIN_BINARY_POS:
        pytest.skip(f"Only {n_pos} hospitalization/fatal positives; need >= {MIN_BINARY_POS}")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_injury_event"] for p in preds])
    auroc  = roc_auc_score(y_true, y_pred)
    print(f"\n  p_injury AUROC: {auroc:.4f}  (target >= {AUROC_P_INJURY_TARGET})")
    assert auroc >= AUROC_P_INJURY_TARGET, (
        f"p_injury AUROC {auroc:.4f} below target {AUROC_P_INJURY_TARGET}"
    )


# ====================================================================== #
#  Test 3b: PR-AUC ratios (signal above naive baseline)
# ====================================================================== #

def test_p_event_pr_auc_ratio():
    """PR-AUC for p_event head must exceed a meaningful lift over the naive baseline.

    When prevalence > 30%, the maximum achievable ratio (1.0 / prevalence) is only
    ~2-3x, so the 4-6x target is physically impossible.  In that case we apply
    an absolute floor: AP >= PR_AUC_AP_FLOOR_EVENT (COVID-adjusted for AUROC~0.76).
    When prevalence <= 30%, we require AP >= PR_AUC_RATIO_P_EVENT * prevalence.
    """
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true     = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    prevalence = float(y_true.mean())
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds      = mt.predict_batch(X)
    y_pred     = np.array([p["p_serious_wr_event"] for p in preds])
    pr_auc     = average_precision_score(y_true, y_pred)
    ratio      = pr_auc / max(prevalence, 1e-9)

    if prevalence > 0.30:
        # High-prevalence: ratio target is physically constrained; use absolute AP floor
        # PR_AUC_AP_FLOOR_EVENT is calibrated to AUROC=0.76 (COVID-adjusted ceiling)
        ap_floor = PR_AUC_AP_FLOOR_EVENT
        print(
            f"\n  p_event PR-AUC={pr_auc:.4f}  prevalence={prevalence:.3f}  ratio={ratio:.2f}x  "
            f"(high-prevalence: AP >= {ap_floor:.3f})"
        )
        assert pr_auc >= ap_floor, (
            f"p_event AP {pr_auc:.4f} below floor {ap_floor:.4f} for prevalence={prevalence:.3f}"
        )
    else:
        print(
            f"\n  p_event PR-AUC={pr_auc:.4f}  prevalence={prevalence:.3f}  "
            f"ratio={ratio:.2f}x  (target >= {PR_AUC_RATIO_P_EVENT}x)"
        )
        assert ratio >= PR_AUC_RATIO_P_EVENT, (
            f"p_event PR-AUC ratio {ratio:.2f}x below target {PR_AUC_RATIO_P_EVENT}x"
        )


def test_p_injury_pr_auc_ratio():
    """PR-AUC for p_injury head must be >= PR_AUC_RATIO_P_INJURY × prevalence."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true     = np.array([r["any_injury_fatal"] for r in rows], dtype=int)
    prevalence = float(y_true.mean())
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient injury/fatal positives")

    preds      = mt.predict_batch(X)
    y_pred     = np.array([p["p_injury_event"] for p in preds])
    pr_auc     = average_precision_score(y_true, y_pred)
    ratio      = pr_auc / max(prevalence, 1e-9)
    print(
        f"\n  p_injury PR-AUC={pr_auc:.4f}  prevalence={prevalence:.3f}  "
        f"ratio={ratio:.2f}x  (target >= {PR_AUC_RATIO_P_INJURY}x)"
    )
    assert ratio >= PR_AUC_RATIO_P_INJURY, (
        f"p_injury PR-AUC ratio {ratio:.2f}x below target {PR_AUC_RATIO_P_INJURY}x"
    )


# ====================================================================== #
#  Test 3c: Brier Skill Score (p_event and p_injury)
# ====================================================================== #

def test_p_event_brier_skill_score():
    """Brier Skill Score for p_event head must be >= BRIER_SS_MIN."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true     = np.array([r["any_wr_serious"] for r in rows], dtype=float)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds      = mt.predict_batch(X)
    y_pred     = np.array([p["p_serious_wr_event"] for p in preds])

    brier      = brier_score_loss(y_true, y_pred)
    prevalence = float(y_true.mean())
    brier_ref  = prevalence * (1.0 - prevalence)  # climatology Brier score
    bss        = 1.0 - brier / max(brier_ref, 1e-9)
    print(
        f"\n  p_event Brier={brier:.4f}  reference={brier_ref:.4f}  "
        f"BSS={bss:.4f}  (target >= {BRIER_SS_MIN})"
    )
    assert bss >= BRIER_SS_MIN, (
        f"p_event Brier Skill Score {bss:.4f} below target {BRIER_SS_MIN}"
    )


def test_p_injury_brier_skill_score():
    """Brier Skill Score for p_injury head must be >= BRIER_SS_MIN."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true     = np.array([r["any_injury_fatal"] for r in rows], dtype=float)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient injury/fatal positives")

    preds      = mt.predict_batch(X)
    y_pred     = np.array([p["p_injury_event"] for p in preds])

    brier      = brier_score_loss(y_true, y_pred)
    prevalence = float(y_true.mean())
    brier_ref  = prevalence * (1.0 - prevalence)
    bss        = 1.0 - brier / max(brier_ref, 1e-9)
    print(
        f"\n  p_injury Brier={brier:.4f}  reference={brier_ref:.4f}  "
        f"BSS={bss:.4f}  (target >= {BRIER_SS_MIN_P_INJURY}; "
        f"theoretical max ~0.10-0.12 at AUROC=0.78, prev=9.5%)"
    )
    assert bss >= BRIER_SS_MIN_P_INJURY, (
        f"p_injury Brier Skill Score {bss:.4f} below target {BRIER_SS_MIN_P_INJURY}"
    )


# ====================================================================== #
#  Test 3d: Expected Calibration Error (ECE)
# ====================================================================== #

def _compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (uniform-width probability bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_pred[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def test_p_event_ece():
    """Expected Calibration Error for p_event head must be < ECE_MAX."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=float)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])
    ece    = _compute_ece(y_true, y_pred)
    print(f"\n  p_event ECE: {ece:.4f}  (max: {ECE_MAX})")
    assert ece < ECE_MAX, f"p_event ECE {ece:.4f} exceeds max {ECE_MAX}"


def test_p_injury_ece():
    """Expected Calibration Error for p_injury head must be < ECE_MAX."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_injury_fatal"] for r in rows], dtype=float)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient injury/fatal positives")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_injury_event"] for p in preds])
    ece    = _compute_ece(y_true, y_pred)
    print(f"\n  p_injury ECE: {ece:.4f}  (max: {ECE_MAX_P_INJURY}; "
          f"note: irreducible ~3.3% floor from train-val prevalence shift)")
    assert ece < ECE_MAX_P_INJURY, f"p_injury ECE {ece:.4f} exceeds max {ECE_MAX_P_INJURY}"


# ====================================================================== #
#  Test 3e: Calibration slope & intercept (logistic regression on logit)
# ====================================================================== #

def _calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray):
    """Fit linear regression of y_true on y_pred (probability scale).

    Returns (slope, intercept) from OLS: E[y_true | y_pred] ≈ slope*y_pred + intercept
    Perfect calibration → slope ≈ 1, intercept ≈ 0.
    """
    X_cal = y_pred.reshape(-1, 1)
    reg   = LinearRegression().fit(X_cal, y_true)
    return float(reg.coef_[0]), float(reg.intercept_)


def test_p_event_calibration_slope():
    """Calibration slope for p_event must be in [CALIB_SLOPE_MIN, CALIB_SLOPE_MAX]."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=float)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds        = mt.predict_batch(X)
    y_pred       = np.array([p["p_serious_wr_event"] for p in preds])
    slope, intercept = _calibration_slope_intercept(y_true, y_pred)
    print(
        f"\n  p_event calibration: slope={slope:.4f}  intercept={intercept:.4f}  "
        f"(targets: slope=[{CALIB_SLOPE_MIN},{CALIB_SLOPE_MAX}]  "
        f"intercept=[{CALIB_INTERCEPT_MIN},{CALIB_INTERCEPT_MAX}])"
    )
    assert CALIB_SLOPE_MIN <= slope <= CALIB_SLOPE_MAX, (
        f"p_event calibration slope {slope:.4f} outside [{CALIB_SLOPE_MIN},{CALIB_SLOPE_MAX}]"
    )
    assert CALIB_INTERCEPT_MIN <= intercept <= CALIB_INTERCEPT_MAX, (
        f"p_event calibration intercept {intercept:.4f} outside "
        f"[{CALIB_INTERCEPT_MIN},{CALIB_INTERCEPT_MAX}]"
    )


def test_p_injury_calibration_slope():
    """Calibration slope for p_injury must be >= CALIB_SLOPE_MIN_P_INJURY.

    The p_injury head has an inherent calibration-slope compression from the
    training-to-validation prevalence shift (~12.8% -> ~9.5%).  Temperature
    scaling corrects the intrinsic slope to ~1.12 on the training holdout, which
    translates to ~0.98 at validation prevalence.  The relaxed lower bound
    (CALIB_SLOPE_MIN_P_INJURY=0.75) accommodates residual shift.
    """
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_injury_fatal"] for r in rows], dtype=float)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient injury/fatal positives")

    preds        = mt.predict_batch(X)
    y_pred       = np.array([p["p_injury_event"] for p in preds])
    slope, intercept = _calibration_slope_intercept(y_true, y_pred)
    print(
        f"\n  p_injury calibration: slope={slope:.4f}  intercept={intercept:.4f}  "
        f"(slope target: [{CALIB_SLOPE_MIN_P_INJURY},{CALIB_SLOPE_MAX_P_INJURY}]  "
        f"intercept target: [{CALIB_INTERCEPT_MIN},{CALIB_INTERCEPT_MAX}])"
    )
    assert CALIB_SLOPE_MIN_P_INJURY <= slope <= CALIB_SLOPE_MAX_P_INJURY, (
        f"p_injury calibration slope {slope:.4f} outside [{CALIB_SLOPE_MIN_P_INJURY},{CALIB_SLOPE_MAX_P_INJURY}]"
    )
    assert CALIB_INTERCEPT_MIN <= intercept <= CALIB_INTERCEPT_MAX, (
        f"p_injury calibration intercept {intercept:.4f} outside "
        f"[{CALIB_INTERCEPT_MIN},{CALIB_INTERCEPT_MAX}]"
    )


# ====================================================================== #
#  Test 4: Regression head — log-penalty Spearman
# ====================================================================== #

def test_log_penalty_spearman():
    """Log-penalty regression head must have Spearman rho >= REGRESSION_SPEARMAN."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    y_true = np.array([r["future_total_penalty"] for r in rows])
    preds  = mt.predict_batch(X)
    y_pred = np.array([p["expected_penalty_usd"] for p in preds])

    # Both on log scale for Spearman stability
    rho, p_val = spearmanr(y_pred, y_true)
    print(f"\n  Log-penalty Spearman rho={rho:.4f}  p={p_val:.4f}")
    assert rho >= REGRESSION_SPEARMAN, (
        f"Log-penalty Spearman {rho:.4f} below minimum {REGRESSION_SPEARMAN}"
    )


# ====================================================================== #
#  Test 5: Regression head — gravity Spearman
# ====================================================================== #

def test_gravity_spearman():
    """Gravity-weighted severity regression head must have Spearman rho >= REGRESSION_SPEARMAN."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    y_true = np.array([r["gravity_weighted_score"] for r in rows])
    preds  = mt.predict_batch(X)
    y_pred = np.array([p["gravity_score"] for p in preds])

    rho, p_val = spearmanr(y_pred, y_true)
    print(f"\n  Gravity Spearman rho={rho:.4f}  p={p_val:.4f}")
    assert rho >= REGRESSION_SPEARMAN, (
        f"Gravity Spearman {rho:.4f} below minimum {REGRESSION_SPEARMAN}"
    )


# ====================================================================== #
#  Test 6: Composite score Spearman (Minimum Viable)
# ====================================================================== #

