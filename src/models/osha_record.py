from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import date

class Violation(BaseModel):
    category: str
    severity: str
    penalty_amount: float
    is_repeat: bool
    is_willful: bool
    description: Optional[str] = None
    gravity: Optional[str] = None
    nr_exposed: Optional[float] = None
    hazardous_substance: Optional[str] = None
    citation_id: Optional[str] = None
    gen_duty_narrative: Optional[str] = None  # inspector plain-language notes (high-priority Gen Duty only)

class AccidentSummary(BaseModel):
    """Accident event linked to an inspection via injury records."""
    summary_nr: str
    event_date: Optional[str] = None
    event_desc: str = ""
    fatality: bool = False
    injuries: List[Dict[str, Any]] = []  # decoded injury dicts
    abstract: str = ""  # pre-joined abstract text (loaded on demand)

class OSHARecord(BaseModel):
    inspection_id: str
    date_opened: date
    violations: List[Violation]
    total_penalties: float
    severe_injury_or_fatality: bool
    accidents: List[AccidentSummary] = []
    naics_code: Optional[str] = None
    nr_in_estab: Optional[str] = None
    estab_name: Optional[str] = None
    site_city: Optional[str] = None
    site_state: Optional[str] = None
