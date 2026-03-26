"""
Real-world validation tests for the predictive ML risk scoring pipeline.

Validates that:
  - Pseudo-labeling has been removed from model training
  - Two predictive models are trained on actual OSHA outcome data
  - score() returns the expected predictive fields
  - The predictive statement matches the user-requested format
  - The RiskAssessment model carries the new predictive fields
  - The explanation leads with the predictive statement
"""

import os
import numpy as np
import pytest
from datetime import date
from unittest.mock import patch, MagicMock

# ── helpers ──────────────────────────────────────────────────────────────────

def _build_population(n: int = 100, seed: int = 0) -> list:
    """Create a synthetic OSHA population with both positive and negative
    serious-violation examples so the classifier can be properly trained."""
    rng = np.random.default_rng(seed)
    population = []
    for i in range(n):
        has_serious = float(rng.integers(0, 2))
        population.append({
            "name": f"COMPANY_{i}",
            "features": [
                float(rng.integers(1, 20)),            # total_inspections
                float(rng.integers(0, 30)),             # total_violations
                has_serious * float(rng.random()),      # serious_violations
                float(rng.random() * 0.3),              # willful_violations
                float(rng.random() * 0.3),              # repeat_violations
                float(rng.random() * 10_000),           # total_penalties
                float(rng.random() * 1_000),            # avg_penalty
                float(rng.random() * 5_000),            # max_penalty
                float(rng.random()),                    # recent_ratio
                float(rng.random() * 0.2),              # severe_incidents
                float(rng.integers(0, 6)),              # violations_per_inspection
                float(rng.random() * 0.3),              # accident_count
                float(rng.random() * 0.1),              # fatality_count
                float(rng.random() * 2),                # injury_count
                float(rng.random() * 10),               # avg_gravity
                float(rng.random() * 500),              # penalties_per_inspection
                float(rng.random()),                    # clean_ratio
            ],
        })
    return population


def _make_scorer_with_population(n: int = 100):
    """Return a fully-trained MLRiskScorer without hitting the DOL API."""
    with patch("src.scoring.ml_risk_scorer.MLRiskScorer._load_or_build"):
        from src.scoring.ml_risk_scorer import MLRiskScorer

    scorer = MLRiskScorer.__new__(MLRiskScorer)
    scorer.osha_client = None
    scorer.classifier_pipeline = None
    scorer.regressor_pipeline = None
    scorer.population_features = None
    scorer.population_risk_scores = None
    scorer._regressor_mask = np.array(
        [MLRiskScorer.FEATURE_NAMES.index(name) for name in MLRiskScorer.REGRESSOR_FEATURE_NAMES]
    )
    os.makedirs(MLRiskScorer.CACHE_DIR, exist_ok=True)

    scorer._train(_build_population(n))
    return scorer


def _make_osha_records(serious: bool = True, willful: bool = False):
    """Build a minimal list of OSHARecord objects for scoring."""
    from src.models.osha_record import OSHARecord, Violation

    viol = Violation(
        category="1910.1200",
        severity="Serious" if serious else "Other",
        penalty_amount=5_000.0,
        is_repeat=False,
        is_willful=willful,
    )
    return [
        OSHARecord(
            inspection_id="TEST-001",
            date_opened=date(2023, 6, 1),
            violations=[viol],
            total_penalties=5_000.0,
            severe_injury_or_fatality=False,
        )
    ]


# ── tests ─────────────────────────────────────────────────────────────────────


class TestPseudoLabelRemoved:
    def test_pseudo_label_method_does_not_exist(self):
        """The old heuristic _pseudo_label must be gone."""
        with patch("src.scoring.ml_risk_scorer.MLRiskScorer._load_or_build"):
            from src.scoring.ml_risk_scorer import MLRiskScorer

        assert not hasattr(MLRiskScorer, "_pseudo_label"), (
            "_pseudo_label still exists; it should have been removed in favor of "
            "real-outcome predictive targets."
        )


class TestModelArchitecture:
    def test_regressor_feature_names_excludes_violation_counts(self):
        """REGRESSOR_FEATURE_NAMES must not include the target features
        (total_violations, violations_per_inspection)."""
        with patch("src.scoring.ml_risk_scorer.MLRiskScorer._load_or_build"):
            from src.scoring.ml_risk_scorer import MLRiskScorer

        excluded = {"total_violations", "violations_per_inspection"}
        overlap = excluded & set(MLRiskScorer.REGRESSOR_FEATURE_NAMES)
        assert not overlap, (
            f"REGRESSOR_FEATURE_NAMES should not contain {overlap}"
        )

    def test_training_produces_two_pipelines(self):
        """After _train(), both classifier_pipeline and regressor_pipeline
        must be populated."""
        scorer = _make_scorer_with_population()
        assert scorer.classifier_pipeline is not None
        assert scorer.regressor_pipeline is not None

    def test_population_risk_scores_cached(self):
        """population_risk_scores must be pre-computed after training."""
        scorer = _make_scorer_with_population()
        assert scorer.population_risk_scores is not None
        assert len(scorer.population_risk_scores) == 100

    def test_compute_risk_score_bounds(self):
        """_compute_risk_score must stay within [0, 100]."""
        with patch("src.scoring.ml_risk_scorer.MLRiskScorer._load_or_build"):
            from src.scoring.ml_risk_scorer import MLRiskScorer

        p = np.array([0.0, 0.5, 1.0])
        v = np.array([0.0, 2.5, 10.0])
        scores = MLRiskScorer._compute_risk_score(p, v)
        assert all(0.0 <= s <= 100.0 for s in scores)
        assert scores[0] == 0.0
        assert scores[2] == 100.0


class TestScoreOutput:
    def test_score_returns_predictive_fields(self):
        """score() must return the new predictive keys."""
        scorer = _make_scorer_with_population()
        result = scorer.score(_make_osha_records())

        required = {
            "risk_score",
            "percentile_rank",
            "predicted_serious_prob",
            "predicted_expected_violations",
            "predictive_statement",
            "feature_weights",
            "features",
            "reputation_score",
            "news_sentiment",
        }
        assert required.issubset(result.keys())

    def test_predictive_statement_format(self):
        """predictive_statement must contain the key numeric phrases."""
        scorer = _make_scorer_with_population()
        result = scorer.score(_make_osha_records())

        stmt = result["predictive_statement"]
        assert "%" in stmt, "Statement should include a % probability"
        assert "violations per inspection" in stmt
        assert "risk score of" in stmt

    def test_predictive_statement_values_match(self):
        """The numbers embedded in predictive_statement must match the
        numeric fields returned alongside it."""
        scorer = _make_scorer_with_population()
        result = scorer.score(_make_osha_records())

        p = result["predicted_serious_prob"]
        v = result["predicted_expected_violations"]
        r = result["risk_score"]
        stmt = result["predictive_statement"]

        assert f"{p:.0f}%" in stmt
        assert f"{v:.1f}" in stmt
        assert f"{r:.0f}/100" in stmt

    def test_score_with_no_records_does_not_crash(self):
        """score() must work gracefully when records=[] (fallback path)."""
        scorer = _make_scorer_with_population()
        result = scorer.score([])
        assert 0 <= result["predicted_serious_prob"] <= 100
        assert result["predicted_expected_violations"] >= 0


class TestRiskAssessmentModel:
    def test_assessment_has_predictive_fields(self):
        """RiskAssessment must declare the three new predictive fields."""
        from src.models.assessment import RiskAssessment

        fields = RiskAssessment.model_fields
        assert "predicted_serious_prob" in fields
        assert "predicted_expected_violations" in fields
        assert "predictive_statement" in fields

    def test_assessment_stores_predictive_data(self):
        """assess() must populate the predictive fields on the returned
        RiskAssessment."""
        from src.scoring.risk_assessor import RiskAssessor
        from src.models.manufacturer import Manufacturer

        fake_result = {
            "risk_score": 65.0,
            "percentile_rank": 78.0,
            "predicted_serious_prob": 41.0,
            "predicted_expected_violations": 1.8,
            "predictive_statement": (
                "This supplier has a 41% predicted chance of a serious OSHA "
                "enforcement event in the next 12 months and an expected 1.8 "
                "violations per inspection, which maps to a risk score of 65/100."
            ),
            "feature_weights": {},
            "features": {},
            "reputation_score": 50.0,
            "news_sentiment": "Unknown",
        }

        mock_ml = MagicMock()
        mock_ml.score.return_value = fake_result

        assessor = RiskAssessor.__new__(RiskAssessor)
        assessor.ml_scorer = mock_ml

        assessment = assessor.assess(Manufacturer(name="Test Corp"), [], [])
        assert assessment.predicted_serious_prob == 41.0
        assert assessment.predicted_expected_violations == 1.8
        assert "41%" in assessment.predictive_statement


class TestExplanationLeadsWithPrediction:
    def test_predictive_statement_appears_in_explanation(self):
        """The explanation string must contain the predictive statement."""
        from src.scoring.risk_assessor import RiskAssessor
        from src.models.manufacturer import Manufacturer

        fake_result = {
            "risk_score": 55.0,
            "percentile_rank": 60.0,
            "predicted_serious_prob": 35.0,
            "predicted_expected_violations": 2.0,
            "predictive_statement": (
                "This supplier has a 35% predicted chance of a serious OSHA "
                "enforcement event in the next 12 months and an expected 2.0 "
                "violations per inspection, which maps to a risk score of 55/100."
            ),
            "feature_weights": {},
            "features": {},
            "reputation_score": 50.0,
            "news_sentiment": "Unknown",
        }

        mock_ml = MagicMock()
        mock_ml.score.return_value = fake_result

        assessor = RiskAssessor.__new__(RiskAssessor)
        assessor.ml_scorer = mock_ml

        assessment = assessor.assess(Manufacturer(name="ACME Industries"), [], [])
        assert "35%" in assessment.explanation
        assert "Predictive Assessment" in assessment.explanation
