"""
grouped_search.py
=================
Entity-grouping and display helpers for the OSHA manufacturer search UI.

Design intent:
  - Users type a parent company name (e.g. "Amazon").
  - OSHA records contain many noisy variants: "Amazon (CMH1)", "Amazon - FWA4", etc.
  - This module groups those variants into a "parent network" with drill-down
    into specific facilities, instead of showing a flat list of raw names.

The grouping layer today is a **heuristic** (normalize + prefix match + token
overlap).  It is intentionally modular so it can be swapped for a proper
entity-resolution backend (e.g. fuzzy matching against an EIN database) later.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process, utils as rfutils
from src.data_retrieval.normalization.company_names import company_match_key as _cmk


SEARCH_SCORE_CUTOFF = 55
SEARCH_LIMIT = 500


# ──────────────────────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FacilityCandidate:
    """One OSHA establishment linked to a grouped company result."""
    raw_name: str               # Exact name as stored in OSHA cache (title-cased)
    display_name: str           # Cleaned-up display string
    facility_code: Optional[str]  # Extracted suffix code e.g. "CMH1", "FWA4"
    city: str
    state: str
    address: str
    naics_code: str             # NAICS industry code from first inspection
    confidence: float           # 0.0–1.0, how likely this variant belongs to the parent
    confidence_label: str       # "High" / "Medium" / "Low"


@dataclass
class GroupedCompanyResult:
    """
    A parent-company search result containing grouped OSHA establishments.
    Returned by group_establishments() for a single search query.
    """
    parent_name: str            # Clean parent label shown to the user
    query: str                  # Original search term entered
    total_facilities: int       # Number of likely-related establishments
    confidence: float           # Overall grouping confidence (0.0–1.0)
    confidence_label: str       # "High" / "Medium" / "Low"
    high_confidence: List[FacilityCandidate] = field(default_factory=list)
    medium_confidence: List[FacilityCandidate] = field(default_factory=list)
    low_confidence: List[FacilityCandidate] = field(default_factory=list)

    @property
    def all_facilities(self) -> List[FacilityCandidate]:
        return self.high_confidence + self.medium_confidence + self.low_confidence

    @property
    def raw_osha_names(self) -> List[str]:
        return [f.raw_name for f in self.all_facilities]

    @property
    def dominant_naics(self) -> str:
        """Most common NAICS code among high- and medium-confidence facilities."""
        codes = [
            f.naics_code for f in self.high_confidence + self.medium_confidence
            if f.naics_code
        ]
        if not codes:
            return ""
        return Counter(codes).most_common(1)[0][0]


@dataclass
class SearchResultSet:
    """
    Full set of results for one search query.

    top_group:      The single best-matched grouped company (if any).
    other_groups:   Additional grouped companies (e.g. "Amazon Logistics" vs "Amazon Fresh").
    unmatched:      Raw OSHA names that matched the search term but couldn't be grouped.
    """
    query: str
    top_group: Optional[GroupedCompanyResult]
    other_groups: List[GroupedCompanyResult]
    unmatched: List[str]


# ──────────────────────────────────────────────────────────────────────────────
#  Company-key search index
#  (persisted to disk so it is built once, not every startup)
#
#  The index is keyed by company_match_key() — the same normalization used
#  to store company_key in the DB — so rapidfuzz matches map directly to DB
#  rows with zero mismatch between search results and scored facilities.
# ──────────────────────────────────────────────────────────────────────────────

_COMPANY_KEY_INDEX_PATH = os.path.join("ml_cache", "company_key_index.json")


def build_company_key_index(osha_client) -> tuple:
    """
    Build the company-key search index from the OSHA database.

    Returns ``(ckey_to_estabs, company_keys)`` where:
      ckey_to_estabs  – dict mapping each company_key (uppercase) to a list of
                        estab dicts: {raw_name, city, state, address, naics_code}
      company_keys    – sorted list of unique company_key strings
    """
    return osha_client.get_company_key_index()


def save_company_key_index(index: tuple) -> None:
    """Persist the company-key index to disk as JSON."""
    ckey_to_estabs, company_keys = index
    os.makedirs(os.path.dirname(_COMPANY_KEY_INDEX_PATH), exist_ok=True)
    with open(_COMPANY_KEY_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"index": ckey_to_estabs, "keys": company_keys}, f)


def load_company_key_index() -> Optional[tuple]:
    """Load a saved company-key index from disk. Returns None if not found."""
    if not os.path.exists(_COMPANY_KEY_INDEX_PATH):
        return None
    with open(_COMPANY_KEY_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["index"], data["keys"]


def get_or_build_company_key_index(osha_client) -> tuple:
    """
    Return the cached company-key index from disk when it is current;
    otherwise build from the DB, save, and return.

    Staleness is checked via a single COUNT(DISTINCT company_key) query.
    """
    cached = load_company_key_index()
    if cached is not None:
        _ckey_to_estabs, company_keys = cached
        if getattr(osha_client, "_use_sqlite", False) and osha_client._db_conn is not None:
            rows = osha_client._db_rows(
                "SELECT COUNT(DISTINCT company_key) AS cnt FROM inspections "
                "WHERE company_key IS NOT NULL AND company_key != ''"
            )
            db_count = rows[0]["cnt"] if rows else 0
            if db_count == len(company_keys):
                return cached
        else:
            return cached
    index = build_company_key_index(osha_client)
    save_company_key_index(index)
    return index


# ──────────────────────────────────────────────────────────────────────────────
#  Name normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Patterns for facility code extraction – captures site codes like CMH1, FWA4,
# KSBD, MQJ1, T5 etc. that appear after separators or in parentheses.
_FACILITY_CODE_RE = re.compile(
    r'(?:[\s\-–]+|\()([A-Z]{1,5}[0-9]{1,4}|[A-Z0-9]{2,8})(?:\)|[\s\-–]|$)',
    re.IGNORECASE,
)

# Corporate / legal suffixes to strip when normalising
_CORP_SUFFIX_RE = re.compile(
    r'[,.]?\s*\b(?:INC\.?|LLC\.?|L\.?P\.?|CORP\.?|CO\.?|LTD\.?|'
    r'LIMITED|INCORPORATED|CORPORATION|COMPANY|GRP|GROUP)\s*$',
    re.IGNORECASE,
)

# Common words that should NOT count as facility codes
_NON_CODE_WORDS = {
    'THE', 'AND', 'LLC', 'INC', 'CORP', 'LTD', 'CO', 'LP',
    'SITE', 'STORE', 'WAREHOUSE', 'FACILITY', 'CENTER', 'CENTRE',
    'MAIN', 'NORTH', 'SOUTH', 'EAST', 'WEST',
    'VISITOR', 'EMPLOYEE', 'ENTRANCE', 'OFFICE', 'AIR', 'LOGISTICS',
}

# Separators between a parent brand and a site suffix
_SEPARATOR_RE = re.compile(r'\s*[-–/|]\s*|\s+@\s+|\s+DBA\s+', re.IGNORECASE)


def normalize_establishment_name(raw: str) -> str:
    """
    Return a cleaned, uppercased name suitable for grouping comparisons.
    Strips branch numbers, corporate suffixes, leading/trailing articles,
    normalises whitespace and '&' ↔ 'AND'.
    """
    s = raw.strip().upper()

    # Strip branch/store prefix like "105891 - "
    s = re.sub(r'^(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,}\s+-\s+', '', s)

    # Strip DBA clause
    s = re.sub(r'\s+D/?B/?A\s+.+$', '', s, flags=re.IGNORECASE)

    # Strip trailing "THE" and leading "THE "
    s = re.sub(r'[,.]?\s*\bTHE\s*$', '', s).strip()
    s = re.sub(r'^\s*THE\s+', '', s).strip()

    # Strip corporate suffixes (up to 3 passes for "CO, INC.")
    for _ in range(3):
        new = _CORP_SUFFIX_RE.sub('', s).strip()
        if new == s:
            break
        s = new

    # Strip dangling punctuation
    s = s.rstrip('.,;-– ').strip()

    # Normalise & → AND
    s = re.sub(r'\s*&\s*', ' AND ', s)
    s = re.sub(r'\s+AND\s+', ' AND ', s, flags=re.IGNORECASE)

    # Collapse whitespace
    s = re.sub(r'\s{2,}', ' ', s).strip()

    return s


def extract_facility_code(raw: str) -> Optional[str]:
    """
    Extract a short alphanumeric facility code from a raw establishment name.

    Examples:
      "Amazon (CMH1)"         → "CMH1"
      "Amazon - FWA4"         → "FWA4"
      "Amazon Air KSBD"       → "KSBD"
      "Amazon - Mqj1 Site"    → "MQJ1"
      "Walmart Store #2341"   → None  (numeric only, treated as store number)
    """
    m = _FACILITY_CODE_RE.search(raw)
    if m:
        code = m.group(1).upper()
        if code not in _NON_CODE_WORDS and not code.isdigit():
            return code
    return None


def score_candidate_match(query_norm: str, candidate_norm: str) -> float:
    """
    Return a confidence score (0.0–1.0) measuring how likely `candidate`
    belongs to the same company as `query`.

    Uses rapidfuzz WRatio which automatically combines token_sort_ratio,
    token_set_ratio, and partial_ratio — handling subset relationships
    (e.g. "WALMART" ↔ "WALMART SUPERCENTER") without manual thresholds.
    """
    q = query_norm.strip()
    c = candidate_norm.strip()
    if not q or not c or len(c) < 2:
        return 0.0
    return fuzz.WRatio(q, c, processor=rfutils.default_process) / 100.0


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    if score >= 0.65:
        return "Medium"
    return "Low"


# ──────────────────────────────────────────────────────────────────────────────
#  Location helpers
# ──────────────────────────────────────────────────────────────────────────────

def _estab_info_for_name(raw_name: str, osha_client) -> Tuple[str, str, str, str]:
    """
    Return (city, state, address, naics_code) for the first inspection associated
    with a raw OSHA establishment name.  Falls back to empty strings gracefully.
    Supports both the SQLite fast path and the in-memory path.
    """
    if getattr(osha_client, "_use_sqlite", False) and osha_client._db_conn is not None:
        # Query by raw estab_name first so each variant gets its own address;
        # fall back to company_key if the raw name doesn't match (e.g. title-case mismatch).
        raw_up = raw_name.strip().upper()
        rows = osha_client._db_rows(
            "SELECT site_city, site_state, site_address, naics_code "
            "FROM inspections WHERE estab_name = ? COLLATE NOCASE LIMIT 1",
            (raw_up,),
        )
        if not rows:
            ck = osha_client.company_match_key(raw_up).upper()
            rows = osha_client._db_rows(
                "SELECT site_city, site_state, site_address, naics_code "
                "FROM inspections WHERE company_key = ? LIMIT 1",
                (ck,),
            )
        if rows:
            r = rows[0]
            return (
                (r.get("site_city",    "") or "").strip().title(),
                (r.get("site_state",   "") or "").strip().upper(),
                (r.get("site_address", "") or "").strip().title(),
                (r.get("naics_code",   "") or "").strip(),
            )
        return "", "", "", ""
    inspections = osha_client._inspections_by_estab.get(raw_name.upper(), [])
    if inspections:
        insp = inspections[0]
        return (
            (insp.get("site_city",    "") or "").strip().title(),
            (insp.get("site_state",   "") or "").strip().upper(),
            (insp.get("site_address", "") or "").strip().title(),
            (insp.get("naics_code",   "") or "").strip(),
        )
    return "", "", "", ""


# ──────────────────────────────────────────────────────────────────────────────
#  Core grouping function
# ──────────────────────────────────────────────────────────────────────────────

def group_establishments(
    query: str,
    company_key_index: tuple,
    osha_client=None,
    max_results: int = 250,  # kept for API compatibility; not used
) -> SearchResultSet:
    """
    Search for a company using the pre-built company-key index.

    The index maps each company_match_key() value (uppercase, already stored in
    the DB ``company_key`` column) to the full list of raw establishment names
    that share that key.  rapidfuzz runs against the compact company_key list
    (not against every raw name), and each hit maps directly to the correct DB
    rows — no secondary expansion or confidence filtering needed.

    This guarantees that the facilities shown in the UI are exactly the same
    set that ``vet_by_raw_estab_names`` will score.
    """
    query_key = _cmk(query)
    if not query_key or len(query_key) < 2:
        return SearchResultSet(query=query, top_group=None, other_groups=[], unmatched=[])

    ckey_to_estabs, company_keys = company_key_index
    if not company_keys:
        return SearchResultSet(query=query, top_group=None, other_groups=[], unmatched=[])

    hits = process.extract(
        query_key, company_keys,
        scorer=fuzz.WRatio,
        processor=rfutils.default_process,
        score_cutoff=SEARCH_SCORE_CUTOFF,
        limit=SEARCH_LIMIT,
    )

    groups: List[GroupedCompanyResult] = []
    unmatched: List[str] = []

    for ckey, rf_score, _idx in hits:
        score = rf_score / 100.0
        estabs = ckey_to_estabs.get(ckey, [])
        if not estabs:
            continue

        facilities = [
            FacilityCandidate(
                raw_name=e["raw_name"],
                display_name=normalize_establishment_name(e["raw_name"]).title(),
                facility_code=extract_facility_code(e["raw_name"]),
                city=e.get("city", ""),
                state=e.get("state", ""),
                address=e.get("address", ""),
                naics_code=e.get("naics_code", ""),
                confidence=score,
                confidence_label=_confidence_label(score),
            )
            for e in estabs
        ]

        # Single low-confidence hit with no clear parent → treat as unmatched
        if len(facilities) == 1 and score < 0.65:
            unmatched.append(facilities[0].raw_name)
            continue

        high = [f for f in facilities if score >= 0.85]
        med  = [f for f in facilities if 0.65 <= score < 0.85]
        low  = [f for f in facilities if score < 0.65]

        groups.append(GroupedCompanyResult(
            parent_name=ckey.title(),
            query=query,
            total_facilities=len(facilities),
            confidence=score,
            confidence_label=_confidence_label(score),
            high_confidence=high,
            medium_confidence=med,
            low_confidence=low,
        ))

    groups.sort(key=lambda g: (-g.confidence, -g.total_facilities))

    return SearchResultSet(
        query=query,
        top_group=groups[0] if groups else None,
        other_groups=groups[1:5],
        unmatched=unmatched[:10],
    )
