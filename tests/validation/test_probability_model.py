"""tests/validation/test_probability_model.py
Tests for the v2 probability-first model: isotonic calibration metrics,
penalty tier head ordering, percentile stretching, confidence signal.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest
from scipy.stats import spearmanr

from tests.validation.mt_shared import (
    MIN_PAIRED,
    MIN_BINARY_POS,
    BRIER_SS_MIN,
    ECE_MAX,
    BRIER_SS_MIN_P_INJURY,
    ECE_MAX_P_INJURY,
    _get_mt_scorer,
    _get_val_data,
)
from src.scoring.calibration import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
)


# ====================================================================== #
#  Isotonic calibration quality — p_serious_wr_event
# ====================================================================== #

def test_brier_skill_score_wr():
    """BSS for isotonic-calibrated p_serious_wr_event >= BRIER_SS_MIN."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data")

    y_wr = np.array([r["any_wr_serious"] for r in rows], dtype=float)
    if y_wr.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds = mt.predict_batch(X)
    p_wr = np.array([p["p_serious_wr_event"] for p in preds])
    bss = brier_skill_score(y_wr, p_wr)
    print(f"\n  WR/Serious BSS: {bss:+.4f}  (minimum: {BRIER_SS_MIN})")
    assert bss >= BRIER_SS_MIN, (
        f"WR/Serious BSS {bss:.4f} below minimum {BRIER_SS_MIN}"
    )


def test_ece_wr():
    """ECE for p_serious_wr_event <= ECE_MAX."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data")

    y_wr = np.array([r["any_wr_serious"] for r in rows], dtype=float)
    if y_wr.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds = mt.predict_batch(X)
    p_wr = np.array([p["p_serious_wr_event"] for p in preds])
    ece = expected_calibration_error(y_wr, p_wr)
    print(f"\n  WR/Serious ECE: {ece:.4f}  (max allowed: {ECE_MAX})")
    assert ece <= ECE_MAX, (
        f"WR/Serious ECE {ece:.4f} above maximum {ECE_MAX}"
    )


# ====================================================================== #
#  Isotonic calibration quality — p_injury_event
# ====================================================================== #

def test_brier_skill_score_injury():
    """BSS for isotonic-calibrated p_injury_event >= BRIER_SS_MIN_P_INJURY."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data")

    y_inj = np.array([r["any_injury_fatal"] for r in rows], dtype=float)
    if y_inj.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds = mt.predict_batch(X)
    p_inj = np.array([p["p_injury_event"] for p in preds])
    bss = brier_skill_score(y_inj, p_inj)
    print(f"\n  Injury BSS: {bss:+.4f}  (minimum: {BRIER_SS_MIN_P_INJURY})")
    assert bss >= BRIER_SS_MIN_P_INJURY, (
        f"Injury BSS {bss:.4f} below minimum {BRIER_SS_MIN_P_INJURY}"
    )


def test_ece_injury():
    """ECE for p_injury_event <= ECE_MAX_P_INJURY."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if not rows:
        pytest.skip("No validation data")

    y_inj = np.array([r["any_injury_fatal"] for r in rows], dtype=float)
    if y_inj.sum() < MIN_BINARY_POS:
        pytest.skip("Insufficient positives")

    preds = mt.predict_batch(X)
    p_inj = np.array([p["p_injury_event"] for p in preds])
    ece = expected_calibration_error(y_inj, p_inj)
    print(f"\n  Injury ECE: {ece:.4f}  (max allowed: {ECE_MAX_P_INJURY})")
    assert ece <= ECE_MAX_P_INJURY, (
        f"Injury ECE {ece:.4f} above maximum {ECE_MAX_P_INJURY}"
    )


# ====================================================================== #
#  Penalty tier head ordering
# ====================================================================== #

def test_penalty_tier_ordering():
    """Mean predicted P(pen>=P75) > P(pen>=P90) > P(pen>=P95) across the population."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    if not hasattr(mt, "_head_pen_p75") or mt._head_pen_p75 is None:
        pytest.skip("Penalty tier heads not trained (old model)")

    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds = mt.predict_batch(X)
    mean_p75 = np.mean([p.get("p_penalty_ge_p75", 0.0) for p in preds])
    mean_p90 = np.mean([p.get("p_penalty_ge_p90", 0.0) for p in preds])
    mean_p95 = np.mean([p.get("p_penalty_ge_p95", 0.0) for p in preds])

    print(f"\n  Penalty tier means: P75={mean_p75:.4f}  P90={mean_p90:.4f}  P95={mean_p95:.4f}")
    assert mean_p75 >= mean_p90, (
        f"P(pen>=P75)={mean_p75:.4f} < P(pen>=P90)={mean_p90:.4f}; tier ordering violated"
    )
    assert mean_p90 >= mean_p95, (
        f"P(pen>=P90)={mean_p90:.4f} < P(pen>=P95)={mean_p95:.4f}; tier ordering violated"
    )


def test_penalty_p95_spearman():
    """P(pen>=P95) should have positive Spearman rho with actual penalty amount."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    if not hasattr(mt, "_head_pen_p95") or mt._head_pen_p95 is None:
        pytest.skip("Penalty tier heads not trained (old model)")

    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    y_pen = np.array([r["future_total_penalty"] for r in rows])
    preds = mt.predict_batch(X)
    p_p95 = np.array([p.get("p_penalty_ge_p95", 0.0) for p in preds])

    rho, _ = spearmanr(p_p95, y_pen)
    print(f"\n  P(pen>=P95) vs actual penalty Spearman: {rho:.4f}")
    assert rho > 0.0, (
        f"P(pen>=P95) has non-positive correlation with actual penalty ({rho:.4f})"
    )


# ====================================================================== #
#  Percentile stretching
# ====================================================================== #

def test_score_range_uses_full_scale():
    """Composite scores should use most of the 0-100 range (not compressed)."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds = mt.predict_batch(X)
    composites = np.array([mt.composite_score(p) for p in preds])
    score_range = composites.max() - composites.min()
    p10 = np.percentile(composites, 10)
    p90 = np.percentile(composites, 90)

    print(f"\n  Score range: [{composites.min():.1f}, {composites.max():.1f}]  "
          f"span={score_range:.1f}  P10={p10:.1f}  P90={p90:.1f}")

    # Scores should span at least 30 points (not all compressed into a narrow band)
    assert score_range >= 30.0, (
        f"Score range {score_range:.1f} too narrow; percentile stretching not working"
    )


def test_score_monotonicity_with_raw():
    """Composite score should be monotonically related to raw composite."""
    mt = _get_mt_scorer()
    if mt is None:
        pytest.skip("Multi-target model not loaded")
    X, rows, _ = _get_val_data()
    if len(rows) < MIN_PAIRED:
        pytest.skip("Insufficient data")

    preds = mt.predict_batch(X)
    raw_scores = np.array([mt._raw_composite(p) for p in preds])
    composites = np.array([mt.composite_score(p) for p in preds])

    rho, _ = spearmanr(raw_scores, composites)
    print(f"\n  Raw vs transformed Spearman: {rho:.4f}")
    assert rho >= 0.95, (
        f"Percentile stretching broke monotonicity (Spearman {rho:.4f} < 0.95)"
    )


# ====================================================================== #
#  Confidence signal
# ====================================================================== #

def test_confidence_detail_structure():
    """Verify _compute_confidence returns expected keys."""
    from src.scoring.risk_assessor import RiskAssessor

    band, detail = RiskAssessor._compute_confidence([], None, None)
    assert band == "low"
    assert "n_inspections" in detail
    assert "recency_years" in detail
    assert "has_mt_model" in detail

    # With some records, should be higher
    from unittest.mock import MagicMock
    from datetime import date

    mock_record = MagicMock()
    mock_record.date_opened = date.today()
    records = [mock_record] * 15

    band2, detail2 = RiskAssessor._compute_confidence(records, {"p_serious_wr_event": 0.5}, None)
    assert band2 in ("high", "medium")
    assert detail2["n_inspections"] == 15
