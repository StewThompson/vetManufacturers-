# src/search/__init__.py
from .grouped_search import (
    GroupedCompanyResult,
    FacilityCandidate,
    SearchResultSet,
    group_establishments,
    normalize_establishment_name,
    extract_facility_code,
    score_candidate_match,
)

__all__ = [
    "GroupedCompanyResult",
    "FacilityCandidate",
    "SearchResultSet",
    "group_establishments",
    "normalize_establishment_name",
    "extract_facility_code",
    "score_candidate_match",
]
