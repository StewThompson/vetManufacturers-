"""
Pydantic response schemas for the REST API.

These are serialisation-safe shapes that mirror the internal models but
use only JSON-native types so FastAPI can encode them without custom
serialisers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


# ── Search ──────────────────────────────────────────────────────────────────

class FacilityOut(BaseModel):
    raw_name: str
    display_name: str
    facility_code: Optional[str]
    city: str
    state: str
    address: str
    naics_code: str
    confidence: float
    confidence_label: str


class GroupedCompanyOut(BaseModel):
    parent_name: str
    total_facilities: int
    confidence: float
    confidence_label: str
    high_confidence: List[FacilityOut]
    medium_confidence: List[FacilityOut]
    low_confidence: List[FacilityOut]


class SearchResponse(BaseModel):
    query: str
    top_group: Optional[GroupedCompanyOut]
    other_groups: List[GroupedCompanyOut]
    unmatched: List[str]


# ── Assessment ───────────────────────────────────────────────────────────────

class ViolationOut(BaseModel):
    category: str
    severity: str
    penalty_amount: float
    is_repeat: bool
    is_willful: bool
    description: Optional[str]
    gravity: Optional[str]
    nr_exposed: Optional[float]
    citation_id: Optional[str]
    gen_duty_narrative: Optional[str]


class AccidentOut(BaseModel):
    summary_nr: str
    event_date: Optional[str]
    event_desc: str
    fatality: bool
    injuries: List[Dict[str, Any]]
    abstract: str


class OSHARecordOut(BaseModel):
    inspection_id: str
    date_opened: str          # ISO date string
    violations: List[ViolationOut]
    total_penalties: float
    severe_injury_or_fatality: bool
    accidents: List[AccidentOut]
    naics_code: Optional[str]
    nr_in_estab: Optional[str]
    estab_name: Optional[str]
    site_city: Optional[str]
    site_state: Optional[str]


class SiteScoreOut(BaseModel):
    name: str
    score: float
    n_inspections: int
    naics_code: Optional[str]
    city: Optional[str]
    state: Optional[str]


class ComplianceOutlook12M(BaseModel):
    """12-month forward compliance projection derived from the risk score and
    per-inspection-rate features."""
    expected_inspections_12m: float
    expected_violations_12m: float
    expected_penalties_usd_12m: int
    expected_serious_12m: float
    expected_willful_repeat_12m: float
    risk_band: Literal["low", "moderate", "high"]
    has_history: bool
    basis: str
    summary_narrative: str


class ProbabilisticRiskTargetsOut(BaseModel):
    """Multi-target probabilistic predictions for the next 12 months."""
    p_serious_wr_event: float
    expected_penalty_usd_12m: float
    expected_citations_12m: float
    p_moderate_penalty_event: float
    p_large_penalty_event: float
    p_extreme_penalty_event: float
    composite_risk_score: float
    large_penalty_threshold_usd: float


class AssessmentResponse(BaseModel):
    manufacturer_name: str
    risk_score: float
    recommendation: Literal["Recommend", "Proceed with Caution", "Do Not Recommend"]
    explanation: str
    confidence_score: float
    feature_weights: Dict[str, float]
    percentile_rank: float
    industry_label: str
    industry_group: str
    industry_percentile: float
    industry_comparison: List[str]
    missing_naics: bool
    establishment_count: int
    site_scores: List[SiteScoreOut]
    risk_concentration: float
    systemic_risk_flag: bool
    aggregation_warning: str
    concentration_warning: str
    records: List[OSHARecordOut]
    record_count: int
    outlook: Optional[ComplianceOutlook12M] = None
    risk_targets: Optional[ProbabilisticRiskTargetsOut] = None


# ── SSE event bodies ─────────────────────────────────────────────────────────

class SSEProgress(BaseModel):
    type: Literal["progress"] = "progress"
    message: str


class SSEResult(BaseModel):
    type: Literal["result"] = "result"
    data: AssessmentResponse


class SSEError(BaseModel):
    type: Literal["error"] = "error"
    message: str
