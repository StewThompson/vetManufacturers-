from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional
from src.models.manufacturer import Manufacturer
from src.models.osha_record import OSHARecord


class ProbabilisticRiskTargets(BaseModel):
    """Multi-target probabilistic predictions for the next 12 months.

    All probabilities are in [0, 1]; monetary values are in USD.
    """
    # Head 1: probability of any Serious / Willful / Repeat event
    p_serious_wr_event: float = 0.0

    # Head 2: expected total OSHA penalty (dollars)
    expected_penalty_usd_12m: float = 0.0

    # Head 3: probability of any hospitalization or fatality event
    p_injury_event: float = 0.0

    # Head 4: gravity-weighted violation severity score (raw, unnormalized)
    gravity_score: float = 0.0

    # Composite score derived from the above (0-100)
    composite_risk_score: float = 0.0


class RiskAssessment(BaseModel):
    manufacturer: Manufacturer
    records: List[OSHARecord]
    risk_score: float  # 0.0 to 100.0
    recommendation: Literal["Recommend", "Proceed with Caution", "Do Not Recommend"]
    explanation: str
    confidence_score: float # 0.0 to 1.0, reflecting data availability/ambiguity
    feature_weights: Dict[str, float] = {}  # ML feature importances
    percentile_rank: float = 50.0  # 0-100, risk percentile among population
    # Industry peer context (populated when NAICS code is available)
    industry_label: str = "Unknown Industry"
    industry_group: str = ""
    industry_percentile: float = 50.0  # percentile within the same NAICS group
    industry_comparison: List[str] = []  # human-readable comparison strings
    missing_naics: bool = False
    # Per-establishment breakdown
    establishment_count: int = 1
    site_scores: List[Dict[str, Any]] = []  # [{name, score, n_inspections, naics_code}]
    risk_concentration: float = 0.0  # fraction of sites scoring ≥ 60
    systemic_risk_flag: bool = False
    aggregation_warning: str = ""
    concentration_warning: str = ""
    # 12-month forward compliance outlook (None when not computable)
    outlook: Optional[Dict[str, Any]] = None
    # Multi-target probabilistic predictions (None when model not loaded)
    risk_targets: Optional[ProbabilisticRiskTargets] = None
