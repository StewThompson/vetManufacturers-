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

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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


def _tokenize(name: str) -> List[str]:
    """Split a normalised name into significant tokens (≥2 chars, not stopwords)."""
    _STOPS = {'AND', 'OF', 'THE', 'A', 'AN', 'FOR', 'IN', 'AT', 'BY', 'TO'}
    return [t for t in re.split(r'\W+', name.upper()) if len(t) >= 2 and t not in _STOPS]


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-like token overlap between two normalised names (0.0–1.0)."""
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def score_candidate_match(query_norm: str, candidate_norm: str) -> float:
    """
    Return a confidence score (0.0–1.0) measuring how likely `candidate`
    belongs to the same company as `query`.

    Scoring:
      1.0  – exact normalised match
      0.85 – candidate starts with query (typical variant pattern)
      0.75 – query starts with candidate (query is more specific)
      0.60 – high token overlap (≥0.8)
      0.45 – medium token overlap (≥0.5)
      0.25 – low token overlap (≥0.3)
      0.0  – no meaningful overlap
    """
    q, c = query_norm.upper(), candidate_norm.upper()

    if q == c:
        return 1.0

    # Strip facility-code suffix from candidate before comparing prefixes
    # e.g. "AMAZON FWA4" → "AMAZON" for prefix comparison
    c_stripped = re.split(r'\s+[-–]?\s*[A-Z]{1,5}[0-9]{1,4}\s*$', c)[0].strip()
    c_stripped = re.sub(r'\s*\([A-Z0-9]{2,8}\)\s*$', '', c_stripped).strip()

    if c.startswith(q) or c_stripped == q:
        return 0.85
    if q.startswith(c):
        return 0.75

    overlap = _token_overlap(q, c)
    if overlap >= 0.8:
        return 0.60
    if overlap >= 0.5:
        return 0.45
    if overlap >= 0.3:
        return 0.25
    return 0.0


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


# ──────────────────────────────────────────────────────────────────────────────
#  Location helpers
# ──────────────────────────────────────────────────────────────────────────────

def _location_info_for_name(raw_name: str, osha_client) -> Tuple[str, str, str]:
    """
    Return (city, state, address) for the first inspection associated with
    a raw OSHA establishment name.  Falls back to empty strings gracefully.
    """
    name_upper = raw_name.upper()
    inspections = osha_client._inspections_by_estab.get(name_upper, [])
    if inspections:
        insp = inspections[0]
        city    = (insp.get("site_city",    "") or "").strip().title()
        state   = (insp.get("site_state",   "") or "").strip().upper()
        address = (insp.get("site_address", "") or "").strip().title()
        return city, state, address
    return "", "", ""


# ──────────────────────────────────────────────────────────────────────────────
#  Core grouping function
# ──────────────────────────────────────────────────────────────────────────────

def group_establishments(
    query: str,
    all_company_names: List[str],
    osha_client,
    max_results: int = 50,
) -> SearchResultSet:
    """
    Given a user search query and the full list of OSHA company names,
    return a SearchResultSet that groups raw names into parent-company buckets.

    Strategy:
      1. Normalise the query.
      2. For every company name that substring-matches the query, compute a
         confidence score.
      3. Cluster names by their normalised prefix (the text before any
         facility-code suffix) to form parent groups.
      4. Rank groups by total high-confidence members.
      5. Return top_group + up to 4 other_groups + unmatched leftovers.
    """
    query_norm = normalize_establishment_name(query)
    query_up   = query_norm.upper()

    if not query_up or len(query_up) < 2:
        return SearchResultSet(query=query, top_group=None, other_groups=[], unmatched=[])

    # ── Step 1: Collect candidates (substring match) ─────────────────────
    candidates: List[Tuple[str, float]] = []   # (raw_name, confidence)
    for name in all_company_names:
        name_norm = normalize_establishment_name(name)
        # Keep names where either direction contains the other
        if query_up not in name_norm.upper() and name_norm.upper() not in query_up:
            # Fallback: token overlap
            score = score_candidate_match(query_norm, name_norm)
            if score < 0.3:
                continue
        else:
            score = score_candidate_match(query_norm, name_norm)
        candidates.append((name, score))

    if not candidates:
        return SearchResultSet(query=query, top_group=None, other_groups=[], unmatched=[])

    # Sort by confidence desc, then alphabetically
    candidates.sort(key=lambda x: (-x[1], x[0]))
    candidates = candidates[:max_results]

    # ── Step 2: Cluster by parent prefix ─────────────────────────────────
    # The "parent prefix" is the normalised name stripped of facility codes.
    def _parent_prefix(name: str) -> str:
        norm = normalize_establishment_name(name).upper()
        # Remove trailing facility code like "FWA4", "CMH1" etc.
        norm = re.sub(r'\s+[-–]?\s*[A-Z]{1,5}[0-9]{1,4}\s*$', '', norm).strip()
        norm = re.sub(r'\s*\([A-Z0-9]{2,8}\)\s*$', '', norm).strip()
        # Remove generic site-descriptor suffixes
        norm = re.sub(
            r'\s+(?:SITE|CAMPUS|FACILITY|WAREHOUSE|DC|FC|IDC|PDC|UNIT|LOC)\s*\d*\s*$',
            '', norm, flags=re.IGNORECASE,
        ).strip()
        # Remove trailing separators
        norm = re.sub(r'\s*[-–/]\s*$', '', norm).strip()
        return norm or normalize_establishment_name(name).upper()

    clusters: Dict[str, List[Tuple[str, float]]] = {}
    for raw, score in candidates:
        prefix = _parent_prefix(raw)
        clusters.setdefault(prefix, []).append((raw, score))

    # ── Step 3: Build GroupedCompanyResult for each cluster ───────────────
    groups: List[GroupedCompanyResult] = []
    unmatched: List[str] = []

    for prefix, members in clusters.items():
        # Determine parent label: the shortest / cleanest member name
        parent_name = min(members, key=lambda x: len(x[0]))[0]
        parent_name = parent_name.title()

        high  = [(r, s) for r, s in members if s >= 0.70]
        med   = [(r, s) for r, s in members if 0.40 <= s < 0.70]
        low   = [(r, s) for r, s in members if s < 0.40]

        # A cluster with only 1 low-confidence member goes to unmatched
        if len(members) == 1 and members[0][1] < 0.40:
            unmatched.append(members[0][0])
            continue

        # Overall group confidence = mean of top-5 member scores
        top_scores = sorted([s for _, s in members], reverse=True)[:5]
        overall_conf = sum(top_scores) / len(top_scores)

        def _build_facility(raw: str, score: float) -> FacilityCandidate:
            city, state, address = _location_info_for_name(raw, osha_client)
            code = extract_facility_code(raw)
            # Build display name: strip facility code from the end for readability
            disp = normalize_establishment_name(raw).title()
            return FacilityCandidate(
                raw_name=raw,
                display_name=disp,
                facility_code=code,
                city=city,
                state=state,
                address=address,
                confidence=score,
                confidence_label=_confidence_label(score),
            )

        groups.append(GroupedCompanyResult(
            parent_name=parent_name,
            query=query,
            total_facilities=len(members),
            confidence=overall_conf,
            confidence_label=_confidence_label(overall_conf),
            high_confidence=[_build_facility(r, s) for r, s in high],
            medium_confidence=[_build_facility(r, s) for r, s in med],
            low_confidence=[_build_facility(r, s) for r, s in low],
        ))

    # Sort groups: prefer higher confidence, more members
    groups.sort(key=lambda g: (-g.confidence, -g.total_facilities))

    top_group    = groups[0] if groups else None
    other_groups = groups[1:5]  # surface up to 4 alternatives

    return SearchResultSet(
        query=query,
        top_group=top_group,
        other_groups=other_groups,
        unmatched=unmatched[:10],
    )
