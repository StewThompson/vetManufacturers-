"""tests/validation/test_regression_heads.py
Regression-head quality gates for the penalty and gravity prediction heads.

Purpose
-------
These tests serve two roles:
  1. **Composite admission gates** — a head must meet the GATE threshold for
     Spearman ρ before its weight in the composite score can be non-zero.
     Failing the gate test means the head is providing noise, not signal.
  2. **Iterative improvement targets** — MINIMUM and STRONG thresholds define
     the performance trajectory we are pushing the model toward.  Strong tests
     failing is a signal to adjust hyperparameters or features.

Thresholds
----------
  Gate     – absolute floor for composite admission.  Head weight MUST be 0
             when the head misses this bar.
  Minimum  – acceptable model quality.  Failing means the head has degraded
             from our last established baseline.
  Strong   – aspirational target that drives iterative improvement.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.scoring.ml_risk_scorer import MLRiskScorer
from src.scoring.multi_target_labeler import build_multi_target_sample
from src.scoring.multi_target_scorer import MultiTargetRiskScorer, _clf_proba_batch, _reg_predict_batch, _apply_platt_batch
from src.scoring.penalty_percentiles import load_percentiles, CACHE_FILENAME as PERCENTILE_CACHE

# ====================================================================== #
#  Thresholds
# ====================================================================== #

CACHE_DIR = "ml_cache"
CUTOFF    = date(2021, 1, 1)
OUTCOME_END = date(2023, 12, 31)
SAMPLE_SIZE = 15_000

# ── Penalty head ──────────────────────────────────────────────────────────
# Spearman ρ (predicted vs actual future total penalty $)
PENALTY_SPEARMAN_GATE    = 0.30   # minimum for composite admission
PENALTY_SPEARMAN_MINIMUM = 0.38   # quality floor (model has degraded if below this)
# Strong target: achievable ceiling with 54 features (Huber-loss hurdle model).
# Improved from baseline 0.352 (scatter plot, pre-improvement) → 0.455+.
# Penalty magnitude is intrinsically noisy (OSHA negotiation, contestation) so
# rho > 0.46 requires new features beyond the current 54-feature set.
PENALTY_SPEARMAN_STRONG  = 0.45   # proven achievable ceiling with current features

# Pearson r in log-space (log1p-transformed both sides) — monetary accuracy
PENALTY_LOG_PEARSON_MIN  = 0.30   # minimum linear signal in log-space

# Top-decile lift: mean(actual penalty | top-10% predicted) / mean(actual penalty)
PENALTY_LIFT_GATE    = 2.5   # minimum for composite admission
PENALTY_LIFT_MINIMUM = 3.5   # quality floor
PENALTY_LIFT_STRONG  = 4.5   # aspirational

# Hurdle model quality
# Mean p(penalty>0) on zero-actual rows must be < nonzero-actual rows (directional)
# Max hurdle false-positive rate: fraction of zero-actual rows with p(pen>0) > 0.5
HURDLE_FPR_MAX = 0.60

# ── Gravity head ──────────────────────────────────────────────────────────
# Spearman ρ (predicted vs actual gravity-weighted severity score)
GRAVITY_SPEARMAN_GATE    = 0.28   # minimum for composite admission
GRAVITY_SPEARMAN_MINIMUM = 0.35   # quality floor
# Strong target: proven achievable ceiling after extensive iteration.
# Improved from baseline 0.330 (scatter plot, pre-improvement) → 0.44+.
# The 40% zero-inflation and outcome-window variance cap this at ~0.44-0.45.
GRAVITY_SPEARMAN_STRONG  = 0.43   # proven achievable with HistGBR + log1p transform

# Pearson r (undifferenced)
GRAVITY_PEARSON_MIN = 0.30

# Top-decile lift
GRAVITY_LIFT_GATE    = 2.0
GRAVITY_LIFT_MINIMUM = 2.8
GRAVITY_LIFT_STRONG  = 3.5

# Zero-row discrimination: mean(pred | actual=0) / mean(pred | actual>0) < threshold
# A perfectly calibrated head would score 0 on zero-actual rows.  We require
# the ratio to be < 0.85 (i.e. model predicts lower risk for zero-outcome rows).
ZERO_PRED_RATIO_MAX = 0.85

# ── Composite gate weight tolerance ──────────────────────────────────────
# If a head's Spearman ρ is below its GATE, the optimised weight must be 0
# (or very close — allow floating-point epsilon 0.01).
GATE_WEIGHT_TOL = 0.01


# ====================================================================== #
#  Data / model singletons (loaded once per session)
# ====================================================================== #

_scorer_singleton    = None
_mt_scorer_singleton = None
_data_singleton      = None


def _get_scorer() -> MLRiskScorer:
    global _scorer_singleton
    if _scorer_singleton is None:
        _scorer_singleton = MLRiskScorer()
    return _scorer_singleton


def _get_mt() -> Optional[MultiTargetRiskScorer]:
    global _mt_scorer_singleton
    if _mt_scorer_singleton is None:
        _mt_scorer_singleton = MultiTargetRiskScorer.load_if_exists(CACHE_DIR)
    return _mt_scorer_singleton


def _get_data():
    """Return (X, predictions, rows) for the 2021-cutoff validation set."""
    global _data_singleton
    if _data_singleton is not None:
        return _data_singleton

    scorer = _get_scorer()
    mt     = _get_mt()
    thresh = load_percentiles(os.path.join(CACHE_DIR, PERCENTILE_CACHE))

    rows = build_multi_target_sample(
        scorer=scorer,
        cutoff_date=CUTOFF,
        outcome_end_date=OUTCOME_END,
        inspections_path=os.path.join(CACHE_DIR, "inspections_bulk.csv"),
        violations_path=os.path.join(CACHE_DIR, "violations_bulk.csv"),
        accidents_path=os.path.join(CACHE_DIR, "accidents_bulk.csv"),
        injuries_path=os.path.join(CACHE_DIR, "accident_injuries_bulk.csv"),
        naics_map=scorer._naics_map,
        penalty_thresholds=thresh,
        sample_size=SAMPLE_SIZE,
    )

    X_raw = np.array([r["features_46"] for r in rows], dtype=float)
    X     = scorer._log_transform_features(np.nan_to_num(X_raw, nan=0.0))
    preds = mt.predict_batch(X)

    _data_singleton = (X, preds, rows)
    return _data_singleton


# ====================================================================== #
#  Fixtures / helpers
# ====================================================================== #

def _get_penalty_arrays():
    """Return (y_actual, y_pred) for future total penalty."""
    X, preds, rows = _get_data()
    y = np.array([r["future_total_penalty"] for r in rows])
    p = np.array([d["expected_penalty_usd"] for d in preds])
    return y, p


def _get_gravity_arrays():
    """Return (y_actual, y_pred) for gravity-weighted severity."""
    X, preds, rows = _get_data()
    y = np.array([r["gravity_weighted_score"] for r in rows])
    p = np.array([d["gravity_score"] for d in preds])
    return y, p


def _top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    """Mean(y | top-10% predicted) / mean(y).  Returns 0 if mean(y)==0."""
    k   = max(1, int(len(p) * 0.10))
    top = np.argsort(p)[::-1][:k]
    base_mean = float(y.mean())
    if base_mean < 1e-9:
        return 0.0
    return float(y[top].mean()) / base_mean


def _skip_if_no_model():
    if _get_mt() is None:
        pytest.skip("MultiTargetRiskScorer not found in ml_cache — run train_multi_target.py first")


def _skip_if_no_data():
    try:
        _get_data()
    except Exception as e:
        pytest.skip(f"Validation data unavailable: {e}")


# ====================================================================== #
#  Penalty Head Tests
# ====================================================================== #

def test_penalty_spearman_gate():
    """Penalty Spearman ρ must be >= GATE to qualify for composite admission."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    rho, _ = spearmanr(p, y)
    print(f"\n  Penalty Spearman ρ = {rho:.4f}  (gate={PENALTY_SPEARMAN_GATE})")
    assert rho >= PENALTY_SPEARMAN_GATE, (
        f"Penalty head Spearman {rho:.4f} below composite-admission gate {PENALTY_SPEARMAN_GATE}.  "
        f"This head has no usable signal; set its composite weight to 0."
    )


def test_penalty_spearman_minimum():
    """Penalty Spearman ρ must be >= MINIMUM (model quality floor)."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    rho, _ = spearmanr(p, y)
    print(f"\n  Penalty Spearman ρ = {rho:.4f}  (minimum={PENALTY_SPEARMAN_MINIMUM})")
    assert rho >= PENALTY_SPEARMAN_MINIMUM, (
        f"Penalty head Spearman {rho:.4f} below minimum {PENALTY_SPEARMAN_MINIMUM}.  "
        f"Model has degraded; check GBR loss function and features."
    )


def test_penalty_spearman_strong():
    """Penalty Spearman ρ must be >= STRONG (iterative improvement target)."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    rho, _ = spearmanr(p, y)
    print(f"\n  Penalty Spearman ρ = {rho:.4f}  (strong={PENALTY_SPEARMAN_STRONG})")
    assert rho >= PENALTY_SPEARMAN_STRONG, (
        f"Penalty head Spearman {rho:.4f} below strong target {PENALTY_SPEARMAN_STRONG}.  "
        f"Iterate on features or model (switch from quantile→huber loss, more trees)."
    )


def test_penalty_log_pearson_minimum():
    """Penalty head Pearson r in log1p-space must be >= minimum.

    Log-space correlation captures monetary magnitude accuracy.
    """
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    r, _ = pearsonr(np.log1p(p), np.log1p(y))
    print(f"\n  Penalty log-Pearson r = {r:.4f}  (minimum={PENALTY_LOG_PEARSON_MIN})")
    assert r >= PENALTY_LOG_PEARSON_MIN, (
        f"Penalty log-space Pearson {r:.4f} below minimum {PENALTY_LOG_PEARSON_MIN}.  "
        f"The model is not capturing the monetary magnitude of future penalties."
    )


def test_penalty_top_decile_lift_gate():
    """Top-10% predicted penalty rows must have >= GATE times higher actual penalty."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    lift = _top_decile_lift(y, p)
    print(f"\n  Penalty top-decile lift = {lift:.2f}x  (gate={PENALTY_LIFT_GATE})")
    assert lift >= PENALTY_LIFT_GATE, (
        f"Penalty top-decile lift {lift:.2f}x below gate {PENALTY_LIFT_GATE}x.  "
        f"High-risk predictions don't concentrate actual high-penalty establishments."
    )


def test_penalty_top_decile_lift_minimum():
    """Penalty top-decile lift must meet the minimum quality floor."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    lift = _top_decile_lift(y, p)
    print(f"\n  Penalty top-decile lift = {lift:.2f}x  (minimum={PENALTY_LIFT_MINIMUM})")
    assert lift >= PENALTY_LIFT_MINIMUM, (
        f"Penalty top-decile lift {lift:.2f}x below minimum {PENALTY_LIFT_MINIMUM}x."
    )


def test_penalty_hurdle_directional():
    """Hurdle p(penalty>0) must be higher for nonzero-actual rows than zero-actual rows.

    This is a basic sanity check: the hurdle classifier must correctly identify
    establishments that will receive penalties.
    """
    _skip_if_no_model()
    _skip_if_no_data()
    X, preds, rows = _get_data()
    mt = _get_mt()
    y = np.array([r["future_total_penalty"] for r in rows])

    head_nz = getattr(mt, "_head_pen_nonzero", None)
    if head_nz is None:
        pytest.skip("Hurdle binary head not found in model")

    raw_hurdle = _clf_proba_batch(head_nz, X)
    platt = getattr(mt, "_platt_pen", np.array([1.0, 0.0]))
    p_any = _apply_platt_batch(raw_hurdle, platt)

    zero_mask    = y == 0
    nonzero_mask = y > 0
    if zero_mask.sum() < 10 or nonzero_mask.sum() < 10:
        pytest.skip("Too few zero/nonzero rows to test directional property")

    mean_zero    = float(p_any[zero_mask].mean())
    mean_nonzero = float(p_any[nonzero_mask].mean())
    print(f"\n  p(pen>0) | zero-actual={mean_zero:.3f}  nonzero-actual={mean_nonzero:.3f}")
    assert mean_nonzero > mean_zero, (
        f"Hurdle model is INVERTED: mean p(pen>0) on zero-actual rows ({mean_zero:.3f}) "
        f">= mean on nonzero-actual rows ({mean_nonzero:.3f}).  "
        f"The hurdle classifier provides no useful discrimination."
    )


def test_penalty_hurdle_fpr():
    """Hurdle false-positive rate must be <= HURDLE_FPR_MAX.

    FPR = fraction of zero-actual-penalty rows where p(pen>0) > 0.5.
    A high FPR means the hurdle assigns penalty probability to establishments
    that will receive no future penalty, inflating predicted values.
    """
    _skip_if_no_model()
    _skip_if_no_data()
    X, preds, rows = _get_data()
    mt = _get_mt()
    y = np.array([r["future_total_penalty"] for r in rows])

    head_nz = getattr(mt, "_head_pen_nonzero", None)
    if head_nz is None:
        pytest.skip("Hurdle binary head not found")

    raw_hurdle = _clf_proba_batch(head_nz, X)
    platt = getattr(mt, "_platt_pen", np.array([1.0, 0.0]))
    p_any = _apply_platt_batch(raw_hurdle, platt)

    zero_mask = y == 0
    if zero_mask.sum() < 10:
        pytest.skip("Too few zero-actual rows")

    fpr = float((p_any[zero_mask] > 0.5).mean())
    print(f"\n  Hurdle FPR on zero-actual rows = {fpr:.1%}  (max={HURDLE_FPR_MAX:.0%})")
    assert fpr <= HURDLE_FPR_MAX, (
        f"Hurdle false-positive rate {fpr:.1%} exceeds maximum {HURDLE_FPR_MAX:.0%}.  "
        f"Model is over-predicting penalties; consider lower max_depth or higher l2_regularization."
    )


# ====================================================================== #
#  Gravity Head Tests
# ====================================================================== #

def test_gravity_spearman_gate():
    """Gravity Spearman ρ must be >= GATE to qualify for composite admission."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    rho, _ = spearmanr(p, y)
    print(f"\n  Gravity Spearman ρ = {rho:.4f}  (gate={GRAVITY_SPEARMAN_GATE})")
    assert rho >= GRAVITY_SPEARMAN_GATE, (
        f"Gravity head Spearman {rho:.4f} below composite-admission gate {GRAVITY_SPEARMAN_GATE}.  "
        f"Set gravity composite weight to 0."
    )


def test_gravity_spearman_minimum():
    """Gravity Spearman ρ must be >= MINIMUM (model quality floor)."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    rho, _ = spearmanr(p, y)
    print(f"\n  Gravity Spearman ρ = {rho:.4f}  (minimum={GRAVITY_SPEARMAN_MINIMUM})")
    assert rho >= GRAVITY_SPEARMAN_MINIMUM, (
        f"Gravity head Spearman {rho:.4f} below minimum {GRAVITY_SPEARMAN_MINIMUM}.  "
        f"Iterate on gravity features or switch to HistGBR with squared_error loss."
    )


def test_gravity_spearman_strong():
    """Gravity Spearman ρ must be >= STRONG (iterative improvement target)."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    rho, _ = spearmanr(p, y)
    print(f"\n  Gravity Spearman ρ = {rho:.4f}  (strong={GRAVITY_SPEARMAN_STRONG})")
    assert rho >= GRAVITY_SPEARMAN_STRONG, (
        f"Gravity head Spearman {rho:.4f} below strong target {GRAVITY_SPEARMAN_STRONG}.  "
        f"Consider log1p-transforming the gravity target during training."
    )


def test_gravity_pearson_minimum():
    """Gravity head Pearson r must be >= minimum (linear correlation)."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    # Use log1p for Pearson to handle the heavy tail
    r, _ = pearsonr(np.log1p(p), np.log1p(y))
    print(f"\n  Gravity log-Pearson r = {r:.4f}  (minimum={GRAVITY_PEARSON_MIN})")
    assert r >= GRAVITY_PEARSON_MIN, (
        f"Gravity log-Pearson {r:.4f} below minimum {GRAVITY_PEARSON_MIN}."
    )


def test_gravity_top_decile_lift_gate():
    """Top-10% predicted gravity rows must have >= GATE times higher actual gravity."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    lift = _top_decile_lift(y, p)
    print(f"\n  Gravity top-decile lift = {lift:.2f}x  (gate={GRAVITY_LIFT_GATE})")
    assert lift >= GRAVITY_LIFT_GATE, (
        f"Gravity top-decile lift {lift:.2f}x below gate {GRAVITY_LIFT_GATE}x."
    )


def test_gravity_top_decile_lift_minimum():
    """Gravity top-decile lift must meet the minimum quality floor."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    lift = _top_decile_lift(y, p)
    print(f"\n  Gravity top-decile lift = {lift:.2f}x  (minimum={GRAVITY_LIFT_MINIMUM})")
    assert lift >= GRAVITY_LIFT_MINIMUM, (
        f"Gravity top-decile lift {lift:.2f}x below minimum {GRAVITY_LIFT_MINIMUM}x."
    )


def test_gravity_zero_row_discrimination():
    """Gravity head must predict lower severity for zero-actual rows.

    mean(pred | gravity_actual == 0) / mean(pred | gravity_actual > 0) < ZERO_PRED_RATIO_MAX.
    """
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()

    zero_mask    = y == 0
    nonzero_mask = y > 0
    if zero_mask.sum() < 10 or nonzero_mask.sum() < 10:
        pytest.skip("Not enough zero/nonzero rows")

    mean_zero    = float(p[zero_mask].mean())
    mean_nonzero = float(p[nonzero_mask].mean())
    if mean_nonzero < 1e-9:
        pytest.skip("Nonzero actual gravity mean is zero — degenerate predictions")

    ratio = mean_zero / mean_nonzero
    print(f"\n  Gravity zero-row pred ratio = {ratio:.3f}  "
          f"(zero={mean_zero:.2f}, nonzero={mean_nonzero:.2f}, max={ZERO_PRED_RATIO_MAX})")
    assert ratio < ZERO_PRED_RATIO_MAX, (
        f"Gravity zero-row ratio {ratio:.3f} >= {ZERO_PRED_RATIO_MAX}.  "
        f"Model predicts nearly equal severity for zero-actual and nonzero-actual rows; "
        f"no useful discrimination."
    )


def test_penalty_zero_row_discrimination():
    """Penalty head must predict lower expected penalty for zero-actual rows."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()

    zero_mask    = y == 0
    nonzero_mask = y > 0
    if zero_mask.sum() < 10 or nonzero_mask.sum() < 10:
        pytest.skip("Not enough zero/nonzero rows")

    mean_zero    = float(p[zero_mask].mean())
    mean_nonzero = float(p[nonzero_mask].mean())
    if mean_nonzero < 1e-9:
        pytest.skip("Nonzero actual penalty mean is zero — degenerate predictions")

    ratio = mean_zero / mean_nonzero
    print(f"\n  Penalty zero-row pred ratio = {ratio:.3f}  "
          f"(zero=${mean_zero:.0f}, nonzero=${mean_nonzero:.0f}, max={ZERO_PRED_RATIO_MAX})")
    assert ratio < ZERO_PRED_RATIO_MAX, (
        f"Penalty zero-row ratio {ratio:.3f} >= {ZERO_PRED_RATIO_MAX}.  "
        f"Model assigns similar predicted penalties to zero- and nonzero-actual rows."
    )


# ====================================================================== #
#  Composite Admission Gate Tests
# ====================================================================== #

def test_composite_gate_penalty_weight():
    """If penalty Spearman < GATE, composite weight for penalty must be 0.

    This test enforces the invariant: the composite score can only incorporate
    a regression head when that head has demonstrated meaningful predictive signal.
    The optimizer should automatically zero out low-signal heads via the
    signal-quality gate logic in _optimize_weights.
    """
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_penalty_arrays()
    mt = _get_mt()

    rho, _ = spearmanr(p, y)
    weights = mt._weights

    # weights[4] = pen_norm component weight
    w_pen = weights[4] if len(weights) > 4 else 0.0
    print(f"\n  Penalty rho={rho:.4f}, composite w_pen={w_pen:.4f}")

    if rho < PENALTY_SPEARMAN_GATE:
        assert w_pen <= GATE_WEIGHT_TOL, (
            f"Penalty head Spearman {rho:.4f} is below gate {PENALTY_SPEARMAN_GATE} "
            f"but composite weight is {w_pen:.4f} (should be ~0).  "
            f"The gate logic in _optimize_weights is not enforcing the admission threshold."
        )
    # If rho >= GATE, any weight is acceptable (optimizer decides)


def test_composite_gate_gravity_weight():
    """If gravity Spearman < GATE, composite weight for gravity must be 0."""
    _skip_if_no_model()
    _skip_if_no_data()
    y, p = _get_gravity_arrays()
    mt = _get_mt()

    rho, _ = spearmanr(p, y)
    weights = mt._weights

    # weights[3] = gravity_norm component weight
    w_grav = weights[3] if len(weights) > 3 else 0.0
    print(f"\n  Gravity rho={rho:.4f}, composite w_grav={w_grav:.4f}")

    if rho < GRAVITY_SPEARMAN_GATE:
        assert w_grav <= GATE_WEIGHT_TOL, (
            f"Gravity head Spearman {rho:.4f} is below gate {GRAVITY_SPEARMAN_GATE} "
            f"but composite weight is {w_grav:.4f} (should be ~0).  "
            f"The gate logic in _optimize_weights is not enforcing the admission threshold."
        )
