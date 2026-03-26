from pydantic import BaseModel
from typing import List, Literal, Dict, Any
from .manufacturer import Manufacturer
from .osha_record import OSHARecord

class RiskAssessment(BaseModel):
    manufacturer: Manufacturer
    records: List[OSHARecord]
    reputation_score: float = 50.0 # 0.0 (Worst) to 100.0 (Best). Default neutral.
    news_sentiment: Literal["Positive", "Neutral", "Negative", "Mixed", "Unknown"] = "Unknown"
    reputation_summary: str = ""
    reputation_data: List[Dict[str, Any]] = [] # Raw news/search results
    risk_score: float  # 0.0 to 100.0
    recommendation: Literal["Recommend", "Proceed with Caution", "Do Not Recommend"]
    explanation: str
    confidence_score: float # 0.0 to 1.0, reflecting data availability/ambiguity
    feature_weights: Dict[str, float] = {}  # ML feature importances
    percentile_rank: float = 50.0  # 0-100, risk percentile among population
    # Predictive model outputs
    predicted_serious_prob: float = 0.0       # 0-100, % chance of serious enforcement event
    predicted_expected_violations: float = 0.0  # expected violations per inspection
    predictive_statement: str = ""             # plain-English predictive summary
