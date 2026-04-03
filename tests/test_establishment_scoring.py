"""
test_establishment_scoring.py — validates the per-establishment scoring refactor.

Tests cover:
1. Single-establishment behaviour (backward compat)
2. Multi-establishment grouping and per-site scoring
3. Recommendation logic (systemic vs. isolated risk)
4. Train/inference feature alignment
5. New API fields present in score() output
"""

import numpy as np
from datetime import date, timedelta

from src.models.osha_record import OSHARecord, Violation


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_violation(severity="Other", penalty=0.0, willful=False, repeat=False, gravity=None):
    return Violation(
        category="1910.147(c)(1)",
        severity=severity,
        penalty_amount=penalty,
        is_repeat=repeat,
        is_willful=willful,
        description=f"Test ({severity})",
        gravity=gravity,
    )


def _make_record(
    inspection_id="INS001",
    date_opened=None,
    violations=None,
    estab_name="TEST FACILITY",
    naics_code="332710",
    accidents=None,
):
    if date_opened is None:
        date_opened = date.today() - timedelta(days=180)
    if violations is None:
        violations = [_make_violation()]
    total_pen = sum(v.penalty_amount for v in violations)
    return OSHARecord(
        inspection_id=inspection_id,
        date_opened=date_opened,
        violations=violations,
        total_penalties=total_pen,
        severe_injury_or_fatality=bool(accidents),
        accidents=accidents or [],
        naics_code=naics_code,
        nr_in_estab="10",
        estab_name=estab_name,
    )


def _get_scorer():
    """Return an MLRiskScorer with a stubbed model (no cache / no API)."""
    from unittest.mock import patch, MagicMock
    from src.scoring.ml_risk_scorer import MLRiskScorer

    fake_pipeline = MagicMock()
    # predict returns 45.0 for any input
    fake_pipeline.predict.return_value = np.array([45.0])
    fake_pipeline.named_steps = {
        "model": MagicMock(
            feature_importances_=np.ones(len(MLRiskScorer.FEATURE_NAMES))
            / len(MLRiskScorer.FEATURE_NAMES)
        )
    }

    with patch.object(MLRiskScorer, "_load_or_build"):
        scorer = MLRiskScorer(osha_client=None)
    scorer.pipeline = fake_pipeline
    scorer.population_features = np.zeros((10, len(MLRiskScorer.FEATURE_NAMES)))
    scorer._industry_stats = {}
    return scorer


# ── 1. Single-establishment behaviour ──────────────────────────────────

def test_single_establishment_returns_one_site():
    scorer = _get_scorer()
    records = [_make_record(estab_name="ACME CORP")]
    result = scorer.score_establishments(records)

    assert result["establishment_count"] == 1
    assert len(result["site_scores"]) == 1
    assert result["site_scores"][0]["name"] == "ACME CORP"
    print("PASS: single-establishment returns 1 site")


def test_single_establishment_score_matches_aggregate():
    scorer = _get_scorer()
    records = [_make_record(estab_name="ACME CORP")]
    result = scorer.score_establishments(records)
    # With one establishment, weighted avg == site score
    assert abs(result["weighted_avg_score"] - result["site_scores"][0]["score"]) < 0.01
    print("PASS: single-site score == weighted avg")


# ── 2. Multi-establishment grouping ───────────────────────────────────

def test_multi_establishment_grouping():
    scorer = _get_scorer()
    records = [
        _make_record(inspection_id="I1", estab_name="PLANT A"),
        _make_record(inspection_id="I2", estab_name="PLANT A"),
        _make_record(inspection_id="I3", estab_name="PLANT B"),
    ]
    result = scorer.score_establishments(records)

    assert result["establishment_count"] == 2
    names = {s["name"] for s in result["site_scores"]}
    assert names == {"PLANT A", "PLANT B"}
    # PLANT A has 2 inspections, PLANT B has 1
    plant_a = next(s for s in result["site_scores"] if s["name"] == "PLANT A")
    plant_b = next(s for s in result["site_scores"] if s["name"] == "PLANT B")
    assert plant_a["n_inspections"] == 2
    assert plant_b["n_inspections"] == 1
    print("PASS: multi-establishment grouping correct")


def test_unknown_estab_name_grouped():
    scorer = _get_scorer()
    records = [
        _make_record(inspection_id="I1", estab_name=None),
        _make_record(inspection_id="I2", estab_name=None),
    ]
    result = scorer.score_establishments(records)

    assert result["establishment_count"] == 1
    assert result["site_scores"][0]["name"] == "UNKNOWN"
    print("PASS: None estab_name grouped as UNKNOWN")


# ── 3. Recommendation logic ───────────────────────────────────────────

def test_recommendation_systemic_high_risk():
    """When >50% of sites are high-risk → systemic → Do Not Recommend."""
    scorer = _get_scorer()
    # Make predict return 70 (high risk) for all
    scorer.pipeline.predict.return_value = np.array([70.0])

    # Each site needs ≥ 5 inspection records so the evidence-gated ceiling
    # (50 for n_insp ≤ 2, 58 for n_insp ≤ 4) does not apply.  The ceiling
    # is 100 at 5+ inspections, allowing the mock score of 70 to pass through.
    records = []
    for site in ("SITE 1", "SITE 2", "SITE 3"):
        for i in range(5):
            records.append(_make_record(inspection_id=f"I{site}{i}", estab_name=site))
    result = scorer.score(records)
    # All 3 sites ≥ 60 → risk_concentration == 1.0 → systemic
    # With score 70 > 60 and systemic → "Do Not Recommend"
    # (Checking via risk_assessor would require more setup, so we check
    #  the ML layer's systemic flag instead)
    estab = scorer.score_establishments(records)
    assert estab["systemic_risk_flag"] is True
    assert estab["risk_concentration"] == 1.0
    print("PASS: systemic high-risk detected")


def test_recommendation_isolated_high_risk():
    """One bad site + two good → not systemic."""
    scorer = _get_scorer()
    call_count = [0]
    def varying_predict(X):
        call_count[0] += 1
        # Return different scores for different establishments
        n = X.shape[0]
        return np.array([70.0 if call_count[0] % 3 == 1 else 20.0] * n)

    scorer.pipeline.predict.side_effect = varying_predict

    records = [
        _make_record(inspection_id="I1", estab_name="BAD SITE"),
        _make_record(inspection_id="I2", estab_name="GOOD SITE 1"),
        _make_record(inspection_id="I3", estab_name="GOOD SITE 2"),
    ]
    estab = scorer.score_establishments(records)
    # Only 1/3 sites high-risk → not systemic
    assert estab["risk_concentration"] < 0.5
    assert estab["systemic_risk_flag"] is False
    print("PASS: isolated high-risk not systemic")


def test_systemic_via_willful_repeat_across_sites():
    """Willful/repeat violations at ≥ 2 sites → systemic."""
    scorer = _get_scorer()
    scorer.pipeline.predict.return_value = np.array([55.0])  # moderate score

    records = [
        _make_record(
            inspection_id="I1", estab_name="SITE A",
            violations=[_make_violation(willful=True, penalty=5000)],
        ),
        _make_record(
            inspection_id="I2", estab_name="SITE B",
            violations=[_make_violation(repeat=True, penalty=3000)],
        ),
    ]
    estab = scorer.score_establishments(records)
    assert estab["systemic_risk_flag"] is True
    print("PASS: willful/repeat across sites → systemic")


# ── 4. Train/inference feature alignment ──────────────────────────────

def test_feature_count_matches():
    scorer = _get_scorer()
    records = [_make_record()]
    X = scorer.extract_features(records)
    assert X.shape == (1, len(scorer.FEATURE_NAMES)), \
        f"Expected (1, {len(scorer.FEATURE_NAMES)}) got {X.shape}"
    print("PASS: feature count matches FEATURE_NAMES")


def test_no_recency_weighting_in_features():
    """Verify that features use simple rates (no exponential decay weighting)."""
    scorer = _get_scorer()
    # Two inspections: one very old, one recent — each with 1 serious violation
    old_record = _make_record(
        inspection_id="OLD",
        date_opened=date.today() - timedelta(days=3650),
        violations=[_make_violation(severity="Serious", penalty=100)],
    )
    new_record = _make_record(
        inspection_id="NEW",
        date_opened=date.today() - timedelta(days=30),
        violations=[_make_violation(severity="Serious", penalty=100)],
    )

    feat17_both, *_ = scorer._extract_establishment_features_raw([old_record, new_record])
    # serious_rate (index 2) should be 2/2 = 1.0 (unweighted)
    assert abs(feat17_both[2] - 1.0) < 1e-9, \
        f"Expected serious_rate=1.0 (simple rate), got {feat17_both[2]}"
    print("PASS: no recency weighting — simple rates used")


def test_extract_features_raw_matches_training_semantics():
    """Feature construction produces rates, matching _fetch_population semantics."""
    scorer = _get_scorer()
    records = [
        _make_record(
            inspection_id="A",
            violations=[
                _make_violation(severity="Serious", penalty=1000),
                _make_violation(severity="Other", penalty=500),
            ],
        ),
        _make_record(
            inspection_id="B",
            violations=[],  # clean inspection
        ),
    ]

    feat17, naics, vpi, avg_pen, sr, wr = scorer._extract_establishment_features_raw(records)

    # n_insp = 2
    assert feat17[0] == 2
    # n_viols = 2
    assert feat17[1] == 2
    # serious_rate = 1/2 = 0.5
    assert abs(feat17[2] - 0.5) < 1e-9
    # vpi = 2/2 = 1.0
    assert abs(feat17[10] - 1.0) < 1e-9
    # clean_ratio = 1/2 = 0.5
    assert abs(feat17[16] - 0.5) < 1e-9
    print("PASS: raw features match training semantics")


# ── 5. New API fields ─────────────────────────────────────────────────

def test_score_returns_new_fields():
    scorer = _get_scorer()
    records = [
        _make_record(inspection_id="I1", estab_name="SITE 1"),
        _make_record(inspection_id="I2", estab_name="SITE 2"),
    ]
    result = scorer.score(records)

    required_keys = {
        "risk_score", "percentile_rank", "feature_weights", "features",
        "industry_label", "industry_group", "industry_percentile",
        "industry_comparison", "missing_naics",
        # New fields:
        "establishment_count", "site_scores", "risk_concentration",
        "systemic_risk_flag", "aggregation_warning", "concentration_warning",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys: {missing}"
    assert result["establishment_count"] == 2
    assert len(result["site_scores"]) == 2
    print("PASS: score() returns all new fields")


def test_aggregation_warning_multi_site():
    scorer = _get_scorer()
    records = [
        _make_record(inspection_id="I1", estab_name="SITE 1"),
        _make_record(inspection_id="I2", estab_name="SITE 2"),
    ]
    result = scorer.score(records)
    assert "aggregates 2 establishments" in result["aggregation_warning"]
    print("PASS: aggregation warning present for multi-site")


def test_no_aggregation_warning_single_site():
    scorer = _get_scorer()
    records = [_make_record(estab_name="SOLO")]
    result = scorer.score(records)
    assert result["aggregation_warning"] == ""
    print("PASS: no aggregation warning for single site")


# ── 6. Weighted average correctness ──────────────────────────────────

def test_weighted_average_calculation():
    """Weighted avg should weight by inspection count."""
    scorer = _get_scorer()

    # Make predict return different scores for different calls
    call_idx = [0]
    scores_to_return = [80.0, 20.0]  # HEAVY has 80, LIGHT has 20

    def mock_predict(X):
        idx = min(call_idx[0], len(scores_to_return) - 1)
        call_idx[0] += 1
        return np.array([scores_to_return[idx]])

    scorer.pipeline.predict.side_effect = mock_predict

    # HEAVY SITE needs ≥ 5 inspection records so the evidence-gated ceiling
    # (58 for n_insp ≤ 4) does not cap the mock score of 80.  At 5+
    # inspections the ceiling is 100, allowing 80 to pass through unchanged.
    records = [
        _make_record(inspection_id="H1", estab_name="HEAVY SITE"),
        _make_record(inspection_id="H2", estab_name="HEAVY SITE"),
        _make_record(inspection_id="H3", estab_name="HEAVY SITE"),
        _make_record(inspection_id="H4", estab_name="HEAVY SITE"),
        _make_record(inspection_id="H5", estab_name="HEAVY SITE"),
        _make_record(inspection_id="L1", estab_name="LIGHT SITE"),
    ]
    estab = scorer.score_establishments(records)

    # HEAVY: 5 inspections × 80 = 400  |  LIGHT: 1 inspection × 20 = 20
    # Weighted avg = 420 / 6 = 70.0
    expected = (80.0 * 5 + 20.0 * 1) / 6
    assert abs(estab["weighted_avg_score"] - expected) < 0.1, \
        f"Expected {expected}, got {estab['weighted_avg_score']}"
    print("PASS: weighted average correct")


# ── 7. OSHARecord estab_name field ────────────────────────────────────

def test_osha_record_has_estab_name():
    r = _make_record(estab_name="MY FACILITY")
    assert r.estab_name == "MY FACILITY"
    print("PASS: OSHARecord.estab_name works")


def test_osha_record_estab_name_optional():
    r = OSHARecord(
        inspection_id="X",
        date_opened=date.today(),
        violations=[],
        total_penalties=0,
        severe_injury_or_fatality=False,
    )
    assert r.estab_name is None
    print("PASS: OSHARecord.estab_name defaults to None")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_single_establishment_returns_one_site,
        test_single_establishment_score_matches_aggregate,
        test_multi_establishment_grouping,
        test_unknown_estab_name_grouped,
        test_recommendation_systemic_high_risk,
        test_recommendation_isolated_high_risk,
        test_systemic_via_willful_repeat_across_sites,
        test_feature_count_matches,
        test_no_recency_weighting_in_features,
        test_extract_features_raw_matches_training_semantics,
        test_score_returns_new_fields,
        test_aggregation_warning_multi_site,
        test_no_aggregation_warning_single_site,
        test_weighted_average_calculation,
        test_osha_record_has_estab_name,
        test_osha_record_estab_name_optional,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) FAILED")
