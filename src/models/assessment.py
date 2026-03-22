from pydantic import BaseModel
from typing import List, Literal, Dict, Any
from src.models.manufacturer import Manufacturer
from src.models.osha_record import OSHARecord

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
    # Industry peer context (populated when NAICS code is available)
    industry_label: str = "Unknown Industry"
    industry_group: str = ""
    industry_percentile: float = 50.0  # percentile within the same NAICS group
    industry_comparison: List[str] = []  # human-readable comparison strings
    missing_naics: bool = False
