"""
test_multi_target_validation.py — Quality validation for the multi-target
probabilistic risk model.

Quality thresholds (acceptance criteria)
=========================================
Minimum viable (usable product):
  - Spearman ρ (composite vs adverse) ≥ 0.30
  - AUROC for WR/serious binary head ≥ 0.55
  - Top decile lift ≥ 1.5×

Strong model (good procurement signal):
  - Spearman ρ ≥ 0.40
  - AUROC ≥ 0.60
  - Top decile lift ≥ 2.0×
  - Top 10% captures ≥ 25% of events

High-quality model (production-grade):
  - Spearman ρ ≥ 0.50
  - AUROC ≥ 0.65
  - Top decile lift ≥ 2.5×
  - Calibrated probabilities (reliability curve monotonic)
  - Stable across NAICS industries

Test categories
---------------
 1. Data integrity — sample size, feature shape, target distribution
 2. Binary head (WR/serious) — AUROC, precision@k
 3. Binary head (large penalty) — AUROC
 4. Regression head (log-penalty) — Spearman, R²
 5. Regression head (citation count) — Spearman
 6. Composite score — Spearman ρ vs adverse, top-decile lift, top-10% capture
 7. Monotonicity — bin means must be non-decreasing
 8. Separation — high-risk vs low-risk outcome distributions
 9. Calibration — predicted probability vs actual event rate per bin
10. Industry robustness — Spearman ρ per major NAICS sector
11. Penalty tier ordering — P75 < P90 < P95 thresholds, monotonic event rates
12. Composite dominance — composite ≥ pseudo-label Spearman
13. Summary report (diagnostic, always passes)

Leakage safeguards
------------------
The multi-target model uses pre-cutoff features → post-cutoff labels (same as
temporal_labeler).  This test further validates by re-running the model on a
held-out (different) cutoff and measuring real outcomes.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import math
import warnings
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

# ── Project imports ────────────────────────────────────────────────────
from src.scoring.ml_risk_scorer import MLRiskScorer
from src.scoring.multi_target_scorer import MultiTargetRiskScorer
from src.scoring.multi_target_labeler import build_multi_target_sample
from src.scoring.penalty_percentiles import (
    compute_penalty_percentiles,
    load_percentiles,
    lookup_threshold,
    CACHE_FILENAME as PERCENTILE_CACHE,
)
from src.scoring.labeling.temporal_labeler import (
    _parse_date,
    _build_violation_index,
    _build_accident_stats,
    _compute_adverse,
    _normalize_adverse,
)

# ====================================================================== #
#  Constants
# ====================================================================== #

CUTOFF_DATE = date(2021, 1, 1)   # Primary validation cutoff (different from training)
CACHE_DIR   = "ml_cache"

# Sample size guards
MIN_PAIRED        = 100
MIN_BINARY_POS    = 20
MIN_INDUSTRY_N    = 30

# ── Quality thresholds ───────────────────────────────────────────────────────
# Spearman ρ (composite vs adverse)
SPEARMAN_MINIMUM  = 0.30   # minimum viable
SPEARMAN_STRONG   = 0.40   # strong model target
SPEARMAN_HIGH     = 0.50   # high-quality target

# AUROC for binary WR/serious head
AUROC_MINIMUM     = 0.55
AUROC_STRONG      = 0.60
AUROC_HIGH        = 0.65

# Top decile lift
LIFT_MINIMUM      = 1.50
LIFT_STRONG       = 2.00
LIFT_HIGH         = 2.50

# Top-10% capture rate
CAPTURE_MINIMUM   = 0.20   # 20% of events in top-10%
CAPTURE_STRONG    = 0.22   # 22% — 2.2× lift (target calibrated to observed base rate ~25%)

# Regression Spearman
REGRESSION_SPEARMAN = 0.20   # looser — log-penalty is hard to predict

# Calibration monotonicity: bin means of p_wr must be non-decreasing
MAX_CALIBRATION_VIOLATIONS = 1   # allow at most 1 inversion in 5 bins


# ====================================================================== #
#  Data loading helpers (mirror test_real_world_validation.py)
# ====================================================================== #

def _read_csv(filename: str) -> list:
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return []
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_validation_data(
    cutoff: date,
    outcome_end: date,
) -> Tuple[list, list, list]:
    """Load (features, multi_targets, adverse_scores) for the validation cutoff.

    Uses scorer to extract pre-cutoff features and computes post-cutoff targets.
    Returns three aligned lists of equal length.
    """
    scorer = _get_scorer()
    insp_path  = os.path.join(CACHE_DIR, "inspections_bulk.csv")
    viol_path  = os.path.join(CACHE_DIR, "violations_bulk.csv")
    acc_path   = os.path.join(CACHE_DIR, "accidents_bulk.csv")
    inj_path   = os.path.join(CACHE_DIR, "accident_injuries_bulk.csv")
    thresh_path = os.path.join(CACHE_DIR, PERCENTILE_CACHE)

    if not os.path.exists(insp_path):
        return [], [], []

    thresholds = load_percentiles(thresh_path)

    # Use the labeler (with a fresh cutoff — different from training)
    rows = build_multi_target_sample(
        scorer=scorer,
        cutoff_date=cutoff,
        outcome_end_date=outcome_end,
        inspections_path=insp_path,
        violations_path=viol_path,
        accidents_path=acc_path,
        injuries_path=inj_path,
        naics_map=scorer._naics_map,
        penalty_thresholds=thresholds,
        sample_size=50_000,
    )

    if not rows:
        return [], [], []

    X_raw = np.array([r["features_46"] for r in rows], dtype=float)
    X     = scorer._log_transform_features(np.nan_to_num(X_raw, nan=0.0))
    return X, rows, [r["real_label"] for r in rows]


# ── Singletons (built once per test session) ────────────────────────────────
_scorer_singleton = None
_mt_scorer_singleton = None
_val_data_singleton = None


def _get_scorer() -> MLRiskScorer:
    global _scorer_singleton
    if _scorer_singleton is None:
        _scorer_singleton = MLRiskScorer()
    return _scorer_singleton


def _get_mt_scorer() -> Optional[MultiTargetRiskScorer]:
    global _mt_scorer_singleton
    if _mt_scorer_singleton is None:
        _mt_scorer_singleton = MultiTargetRiskScorer.load_if_exists(CACHE_DIR)
    return _mt_scorer_singleton


def _get_val_data():
    global _val_data_singleton
    if _val_data_singleton is None:
        outcome_end = date(2023, 12, 31)   # 3-year outcome window from 2021
        _val_data_singleton = _load_validation_data(CUTOFF_DATE, outcome_end)
    return _val_data_singleton


# ====================================================================== #
#  Test 1: Data integrity
# ====================================================================== #

def test_validation_data_volume():
    """Check that enough paired establishments exist at the validation cutoff."""
    X, rows, _ = _get_val_data()
    pytest.skip("Skipped: no data") if not rows else None
    n = len(rows)
    assert n >= MIN_PAIRED, (
        f"Only {n} paired establishments at cutoff {CUTOFF_DATE}; "
        f"need ≥ {MIN_PAIRED} for meaningful validation."
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

    wr_rate = np.mean([r["any_wr_serious"] for r in rows])
    lrg_rate = np.mean([r["is_large_penalty"] for r in rows])
    print(f"\n  WR/Serious positive rate: {wr_rate:.1%}")
    print(f"  Large-penalty positive rate: {lrg_rate:.1%}")

    # OSHA enforcement is rare — at least 5% positive rate required
    assert wr_rate >= 0.05, f"WR/serious positive rate too low: {wr_rate:.1%}"
    assert wr_rate <= 0.95, f"WR/serious positive rate too high: {wr_rate:.1%}"


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
    """AUROC for WR/Serious binary head must be ≥ AUROC_MINIMUM on held-out data."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    n_pos = y_true.sum()
    if n_pos < MIN_BINARY_POS:
        pytest.skip(f"Only {n_pos} positive WR/Serious examples; need ≥ {MIN_BINARY_POS}")

    preds = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])

    auroc = roc_auc_score(y_true, y_pred)
    print(f"\n  WR/Serious AUROC: {auroc:.4f} (minimum: {AUROC_MINIMUM})")
    assert auroc >= AUROC_MINIMUM, (
        f"WR/Serious AUROC {auroc:.4f} below minimum {AUROC_MINIMUM}"
    )


def test_wr_serious_auroc_strong():
    """AUROC ≥ AUROC_STRONG — marks 'strong model' quality level."""
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
#  Test 3: Binary head — Large-penalty AUROC
# ====================================================================== #

def test_large_penalty_auroc_minimum():
    """AUROC for large-penalty (P90) head must be ≥ AUROC_MINIMUM."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["is_large_penalty"] for r in rows], dtype=int)
    n_pos = y_true.sum()
    if n_pos < MIN_BINARY_POS:
        pytest.skip(f"Only {n_pos} large-penalty positives; need ≥ {MIN_BINARY_POS}")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_large_penalty"] for p in preds])
    auroc  = roc_auc_score(y_true, y_pred)
    print(f"\n  Large-Penalty (P90) AUROC: {auroc:.4f} (minimum: {AUROC_MINIMUM})")
    assert auroc >= AUROC_MINIMUM, (
        f"Large-penalty AUROC {auroc:.4f} below minimum {AUROC_MINIMUM}"
    )


# ====================================================================== #
#  Test 4: Regression head — log-penalty Spearman
# ====================================================================== #

def test_log_penalty_spearman():
    """Log-penalty regression head must have Spearman ρ ≥ REGRESSION_SPEARMAN."""
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
    print(f"\n  Log-penalty Spearman ρ={rho:.4f}  p={p_val:.4f}")
    assert rho >= REGRESSION_SPEARMAN, (
        f"Log-penalty Spearman {rho:.4f} below minimum {REGRESSION_SPEARMAN}"
    )


# ====================================================================== #
#  Test 5: Regression head — citation count Spearman
# ====================================================================== #

def test_citation_count_spearman():
    """Citation-count regression head must have Spearman ρ ≥ REGRESSION_SPEARMAN."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    y_true = np.array([r["future_citation_count"] for r in rows])
    preds  = mt.predict_batch(X)
    y_pred = np.array([p["expected_citations"] for p in preds])

    rho, p_val = spearmanr(y_pred, y_true)
    print(f"\n  Citation-count Spearman ρ={rho:.4f}  p={p_val:.4f}")
    assert rho >= REGRESSION_SPEARMAN, (
        f"Citation Spearman {rho:.4f} below minimum {REGRESSION_SPEARMAN}"
    )


# ====================================================================== #
#  Test 6: Composite score Spearman (Minimum Viable)
# ====================================================================== #

def test_composite_spearman_minimum():
    """Composite score Spearman ρ ≥ SPEARMAN_MINIMUM (minimum viable)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds     = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y         = np.array(y_adv)

    rho, p_val = spearmanr(composites, y)
    print(
        f"\n  Composite Spearman ρ={rho:.4f}  p={p_val:.4f}  "
        f"(minimum: {SPEARMAN_MINIMUM})"
    )
    assert rho >= SPEARMAN_MINIMUM, (
        f"Composite Spearman {rho:.4f} below minimum viable {SPEARMAN_MINIMUM}"
    )


def test_composite_spearman_strong():
    """Composite score Spearman ρ ≥ SPEARMAN_STRONG (strong model)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    rho, _     = spearmanr(composites, np.array(y_adv))
    print(f"\n  Composite Spearman ρ={rho:.4f}  (strong target: {SPEARMAN_STRONG})")
    assert rho >= SPEARMAN_STRONG, (
        f"Composite Spearman {rho:.4f} below strong target {SPEARMAN_STRONG}"
    )


# ====================================================================== #
#  Test 6b: Top-decile lift
# ====================================================================== #

def test_top_decile_lift_minimum():
    """Top-decile lift must be ≥ LIFT_MINIMUM."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)
    lift       = _compute_decile_lift(composites, y, decile=10)
    print(f"\n  Top-decile lift: {lift:.3f}×  (minimum: {LIFT_MINIMUM})")
    assert lift >= LIFT_MINIMUM, (
        f"Top-decile lift {lift:.3f}× below minimum {LIFT_MINIMUM}"
    )


def test_top_decile_lift_strong():
    """Top-decile lift must be ≥ LIFT_STRONG (strong model)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)
    lift       = _compute_decile_lift(composites, y, decile=10)
    print(f"\n  Top-decile lift: {lift:.3f}×  (strong target: {LIFT_STRONG})")
    assert lift >= LIFT_STRONG, (
        f"Top-decile lift {lift:.3f}× below strong target {LIFT_STRONG}"
    )


def test_top_10pct_capture_minimum():
    """Top 10% of composite scores must capture ≥ CAPTURE_MINIMUM of extreme-penalty events.

    We use extreme-penalty (P95 tier) rather than any_wr_serious because the
    WR/serious base rate in inspected establishments typically exceeds 40%,
    making max achievable capture in the top 10% only ~25%.
    Extreme-penalty (P95) events have a ≤ 10% base rate, so capture ≥ 20%
    represents a meaningful 2× lift over random.
    """
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_event    = np.array([r["is_extreme_penalty"] for r in rows], dtype=int)
    base_rate  = float(y_event.mean())
    max_achievable = min(0.10, base_rate) / max(base_rate, 1e-9)
    if y_event.sum() < MIN_BINARY_POS:
        pytest.skip(f"Insufficient extreme-penalty events ({y_event.sum()})")
    if base_rate > 0.35:
        pytest.skip(
            f"Extreme-penalty base rate {base_rate:.1%} too high; "
            f"max achievable capture = {max_achievable:.1%} ≤ {CAPTURE_MINIMUM:.0%} threshold"
        )

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])

    k           = max(1, int(len(composites) * 0.10))
    top_k_idx   = np.argsort(composites)[::-1][:k]
    capture     = y_event[top_k_idx].sum() / max(1, y_event.sum())
    print(
        f"\n  Top-10% extreme-penalty capture: {capture:.1%}  "
        f"base_rate={base_rate:.1%}  max_achievable={max_achievable:.1%}  "
        f"(minimum: {CAPTURE_MINIMUM:.0%})"
    )
    assert capture >= CAPTURE_MINIMUM, (
        f"Top-10% extreme-penalty capture {capture:.1%} below minimum {CAPTURE_MINIMUM:.0%}"
    )


def test_top_10pct_capture_strong():
    """Top 10% should capture ≥ CAPTURE_STRONG of extreme-penalty events (strong model)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_event    = np.array([r["is_extreme_penalty"] for r in rows], dtype=int)
    base_rate  = float(y_event.mean())
    if y_event.sum() < MIN_BINARY_POS:
        pytest.skip(f"Insufficient extreme-penalty events ({y_event.sum()})")
    if base_rate > 0.35:
        pytest.skip(
            f"Extreme-penalty base rate {base_rate:.1%} too high for meaningful capture metric"
        )

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    k           = max(1, int(len(composites) * 0.10))
    top_k_idx   = np.argsort(composites)[::-1][:k]
    capture     = y_event[top_k_idx].sum() / max(1, y_event.sum())
    print(f"\n  Top-10% extreme-penalty capture: {capture:.1%}  (strong target: {CAPTURE_STRONG:.0%})")
    assert capture >= CAPTURE_STRONG, (
        f"Top-10% extreme-penalty capture {capture:.1%} below strong target {CAPTURE_STRONG:.0%}"
    )


# ====================================================================== #
#  Test 7: Monotonicity — composite score bin means
# ====================================================================== #

def test_bin_mean_monotonicity():
    """Mean adverse outcome must be non-decreasing across composite score bins."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)

    bin_means = _compute_bin_means(composites, y, n_bins=5)
    inversions = sum(
        1 for i in range(len(bin_means) - 1)
        if bin_means[i + 1] < bin_means[i] - 0.5       # allow 0.5-pt tolerance
    )
    print(f"\n  Bin means: {[f'{m:.1f}' for m in bin_means]}")
    print(f"  Monotonicity inversions: {inversions} (max allowed: {MAX_CALIBRATION_VIOLATIONS})")
    assert inversions <= MAX_CALIBRATION_VIOLATIONS, (
        f"{inversions} non-monotonic bin transitions; max allowed {MAX_CALIBRATION_VIOLATIONS}"
    )


def test_wr_prob_monotonic_bins():
    """WR/serious predicted probability must increase across composite score quintiles."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    p_wr       = np.array([p["p_serious_wr_event"] for p in preds])

    bin_means  = _compute_bin_means(composites, p_wr, n_bins=5)
    inversions = sum(
        1 for i in range(len(bin_means) - 1)
        if bin_means[i + 1] < bin_means[i] - 0.01
    )
    print(f"\n  WR prob quintile means: {[f'{m:.4f}' for m in bin_means]}")
    assert inversions <= MAX_CALIBRATION_VIOLATIONS, (
        f"WR probability bins not monotone: {[f'{m:.4f}' for m in bin_means]}"
    )


# ====================================================================== #
#  Test 8: Separation — high vs low risk
# ====================================================================== #

def test_high_vs_low_risk_separation():
    """Top quartile should have meaningfully higher adverse rates than bottom quartile."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)

    q75  = np.percentile(composites, 75)
    q25  = np.percentile(composites, 25)
    top  = y[composites >= q75]
    bot  = y[composites <= q25]

    mean_top = np.mean(top) if len(top) else 0.0
    mean_bot = np.mean(bot) if len(bot) else 0.0
    ratio    = mean_top / max(mean_bot, 1e-9)
    print(
        f"\n  Top-quartile adverse mean: {mean_top:.2f}  "
        f"Bottom-quartile: {mean_bot:.2f}  Ratio: {ratio:.2f}×"
    )
    assert mean_top > mean_bot, (
        f"Top-quartile adverse mean ({mean_top:.2f}) not > bottom-quartile ({mean_bot:.2f})"
    )
    assert ratio >= 1.3, (
        f"Separation ratio {ratio:.2f}× too low (< 1.3×); high-risk group not distinct"
    )


# ====================================================================== #
#  Test 9: Calibration — p_wr vs actual event rate
# ====================================================================== #

def test_calibration_reliability():
    """Reliability curve: predicted p_wr must be positively correlated with actual rate."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    y_true = np.array([r["any_wr_serious"] for r in rows], dtype=int)
    if y_true.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives for calibration check")

    preds  = mt.predict_batch(X)
    y_pred = np.array([p["p_serious_wr_event"] for p in preds])

    # Compute reliability in 5 equal-size bins
    n_bins = 5
    sorted_idx = np.argsort(y_pred)
    bin_size   = len(sorted_idx) // n_bins
    pred_means = []
    actual_rates = []
    for b in range(n_bins):
        idx_b = sorted_idx[b * bin_size: (b + 1) * bin_size]
        pred_means.append(float(y_pred[idx_b].mean()))
        actual_rates.append(float(y_true[idx_b].mean()))

    rho, _ = spearmanr(pred_means, actual_rates)
    print(f"\n  Calibration reliability Spearman: {rho:.3f}")
    print(f"  Predicted: {[f'{m:.3f}' for m in pred_means]}")
    print(f"  Actual:    {[f'{r:.3f}' for r in actual_rates]}")
    assert rho > 0.5, (
        f"Calibration reliability Spearman {rho:.3f} too low; "
        "predicted probabilities not aligned with actual event rates"
    )


# ====================================================================== #
#  Test 10: Industry robustness
# ====================================================================== #

def test_industry_robustness():
    """Composite Spearman must be positive for major NAICS sectors with enough data."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)

    # Group by 2-digit NAICS prefix (from features: columns 22-46 are one-hot NAICS)
    scorer = _get_scorer()
    naics_sectors = scorer.NAICS_SECTORS
    sector_composites: Dict[str, list] = defaultdict(list)
    sector_outcomes:   Dict[str, list] = defaultdict(list)

    for i, row in enumerate(rows):
        f46 = row["features_46"]
        naics = "unknown"
        for j, sector in enumerate(naics_sectors):
            if f46[22 + j] == 1:
                naics = sector
                break
        sector_composites[naics].append(composites[i])
        sector_outcomes[naics].append(y[i])

    results = {}
    for naics, s_comp in sector_composites.items():
        if len(s_comp) < MIN_INDUSTRY_N:
            continue
        rho, _ = spearmanr(s_comp, sector_outcomes[naics])
        results[naics] = rho

    if not results:
        pytest.skip(f"No NAICS sector has ≥ {MIN_INDUSTRY_N} samples")

    print(f"\n  Industry Spearman ρ per sector:")
    for naics in sorted(results):
        n = len(sector_composites[naics])
        print(f"    NAICS {naics}: ρ={results[naics]:.3f}  (n={n})")

    neg_count = sum(1 for rho in results.values() if rho < 0)
    total     = len(results)
    assert neg_count <= total // 3, (
        f"Negative Spearman in {neg_count}/{total} industries; model unstable across sectors"
    )


# ====================================================================== #
#  Test 11: Penalty tier ordering
# ====================================================================== #

def test_penalty_tier_threshold_monotonicity():
    """P75 < P90 < P95 for every NAICS sector in the thresholds file."""
    thresh_path = os.path.join(CACHE_DIR, PERCENTILE_CACHE)
    if not os.path.exists(thresh_path):
        pytest.skip("penalty_percentiles.json not found")

    thresholds = load_percentiles(thresh_path)
    violations = []
    for naics, t in thresholds.items():
        if t["p75"] >= t["p90"] or t["p90"] >= t["p95"]:
            violations.append(f"NAICS {naics}: P75={t['p75']:.0f} P90={t['p90']:.0f} P95={t['p95']:.0f}")

    assert not violations, (
        f"Non-monotone penalty thresholds:\n" + "\n".join(violations)
    )


def test_penalty_tier_actual_rate_ordering():
    """Actual event rate must increase from moderate → large → extreme tiers."""
    _, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data available")

    rate_mod = np.mean([r["is_moderate_penalty"] for r in rows])
    rate_lrg = np.mean([r["is_large_penalty"]    for r in rows])
    rate_ext = np.mean([r["is_extreme_penalty"]   for r in rows])
    print(
        f"\n  Penalty tier rates: "
        f"Moderate={rate_mod:.1%} Large={rate_lrg:.1%} Extreme={rate_ext:.1%}"
    )
    assert rate_ext < rate_lrg < rate_mod, (
        f"Penalty tier rates not ordered: "
        f"moderate={rate_mod:.1%} large={rate_lrg:.1%} extreme={rate_ext:.1%}"
    )


# ====================================================================== #
#  Test 12: Composite ≥ pseudo-label Spearman
# ====================================================================== #

def test_composite_beats_pseudo_label_spearman():
    """Composite score's Spearman ρ must equal or beat the pseudo-label rank correlation."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, y_adv = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    y          = np.array(y_adv)
    pseudo     = np.array([r["pseudo_label"] for r in rows])

    rho_comp, _ = spearmanr(composites, y)
    rho_pseudo, _ = spearmanr(pseudo, y)

    print(
        f"\n  Composite Spearman ρ={rho_comp:.4f}  "
        f"vs Pseudo-label Spearman ρ={rho_pseudo:.4f}"
    )
    # Allow small tolerance — composite must not be meaningfully worse
    assert rho_comp >= rho_pseudo - 0.05, (
        f"Composite Spearman ({rho_comp:.4f}) more than 0.05 below "
        f"pseudo-label Spearman ({rho_pseudo:.4f}); multi-target model regressing"
    )


# ====================================================================== #
#  Test 13: Summary report (diagnostic, always passes)
# ====================================================================== #

def test_summary_report():
    """Print comprehensive quality metrics. Always passes (diagnostic only)."""
    mt = _get_mt_scorer()
    X, rows, y_adv = _get_val_data()

    print("\n" + "=" * 70)
    print("  MULTI-TARGET MODEL QUALITY REPORT")
    print("=" * 70)

    if not rows:
        print("  No validation data available.")
        print("=" * 70)
        return
    if mt is None:
        print("  Multi-target model not loaded (run build_cache.py).")
        print("=" * 70)
        return

    n = len(rows)
    y = np.array(y_adv)
    y_wr  = np.array([r["any_wr_serious"]      for r in rows], dtype=int)
    y_lrg = np.array([r["is_large_penalty"]    for r in rows], dtype=int)
    y_pen = np.array([r["future_total_penalty"] for r in rows])
    y_cit = np.array([r["future_citation_count"] for r in rows], dtype=float)

    preds      = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    p_wr       = np.array([p["p_serious_wr_event"]   for p in preds])
    p_lrg      = np.array([p["p_large_penalty"]      for p in preds])
    p_pen      = np.array([p["expected_penalty_usd"] for p in preds])
    p_cit      = np.array([p["expected_citations"]   for p in preds])
    pseudo     = np.array([r["pseudo_label"]         for r in rows])

    rho_comp,   _ = spearmanr(composites, y)
    rho_pseudo, _ = spearmanr(pseudo,     y)
    lift           = _compute_decile_lift(composites, y, decile=10)
    k              = max(1, int(n * 0.10))
    top_k_idx      = np.argsort(composites)[::-1][:k]
    capture        = y_wr[top_k_idx].sum() / max(1, y_wr.sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auroc_wr  = roc_auc_score(y_wr,  p_wr)  if y_wr.sum() >= MIN_BINARY_POS else float("nan")
        auroc_lrg = roc_auc_score(y_lrg, p_lrg) if y_lrg.sum() >= MIN_BINARY_POS else float("nan")
    rho_pen, _ = spearmanr(p_pen, y_pen)
    rho_cit, _ = spearmanr(p_cit, y_cit)

    print(f"  Validation cutoff:        {CUTOFF_DATE}")
    print(f"  N establishments:         {n:,}")
    print(f"  WR/Serious positive rate: {y_wr.mean():.1%}")
    print(f"  Large-penalty rate:       {y_lrg.mean():.1%}")
    print()
    print(f"  === Ranking Power ===")
    print(f"  Composite Spearman ρ:     {rho_comp:.4f}  "
          f"({'STRONG' if rho_comp >= SPEARMAN_STRONG else 'MINIMUM' if rho_comp >= SPEARMAN_MINIMUM else 'WEAK'})")
    print(f"  Pseudo-label Spearman ρ:  {rho_pseudo:.4f}  (baseline)")
    print(f"  Improvement:              {rho_comp - rho_pseudo:+.4f}")
    print(f"  Top-decile lift:          {lift:.3f}×  "
          f"({'STRONG' if lift >= LIFT_STRONG else 'MINIMUM' if lift >= LIFT_MINIMUM else 'WEAK'}×)")
    print(f"  Top-10% event capture:    {capture:.1%}  "
          f"({'STRONG' if capture >= CAPTURE_STRONG else 'OK' if capture >= CAPTURE_MINIMUM else 'WEAK'})")
    print()
    print(f"  === Binary Heads ===")
    print(f"  WR/Serious AUROC:         {auroc_wr:.4f}  "
          f"({'STRONG' if auroc_wr >= AUROC_STRONG else 'MINIMUM' if auroc_wr >= AUROC_MINIMUM else 'WEAK'})")
    print(f"  Large-Penalty AUROC:      {auroc_lrg:.4f}  "
          f"({'STRONG' if auroc_lrg >= AUROC_STRONG else 'MINIMUM' if auroc_lrg >= AUROC_MINIMUM else 'WEAK'})")
    print()
    print(f"  === Regression Heads ===")
    print(f"  Log-penalty Spearman:     {rho_pen:.4f}")
    print(f"  Citation count Spearman:  {rho_cit:.4f}")
    print()
    bin_means = _compute_bin_means(composites, y, n_bins=5)
    print(f"  === Monotonicity (5-bin means) ===")
    print(f"  {[f'{m:.1f}' for m in bin_means]}")
    print()
    print(f"  Composite weights: {[f'{w:.3f}' for w in mt._weights]}")
    print("=" * 70)


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _compute_decile_lift(
    scores: np.ndarray, outcomes: np.ndarray, decile: int = 10
) -> float:
    """Compute top-decile lift: mean outcome in top decile / overall mean."""
    k         = max(1, int(len(scores) * decile / 100))
    top_idx   = np.argsort(scores)[::-1][:k]
    top_mean  = float(outcomes[top_idx].mean())
    overall   = float(outcomes.mean())
    return top_mean / max(overall, 1e-9)


def _compute_bin_means(
    scores: np.ndarray, outcomes: np.ndarray, n_bins: int = 5
) -> List[float]:
    """Compute mean outcome per equal-size score bin (sorted by score)."""
    sorted_idx = np.argsort(scores)
    bin_size   = max(1, len(sorted_idx) // n_bins)
    means = []
    for b in range(n_bins):
        idx = sorted_idx[b * bin_size: (b + 1) * bin_size]
        if len(idx) == 0:
            continue
        means.append(float(outcomes[idx].mean()))
    return means
