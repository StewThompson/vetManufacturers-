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
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process, utils as rfutils


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
#  Pre-built name index (persisted to disk so it's built once, not every startup)
# ──────────────────────────────────────────────────────────────────────────────

import json
import os

_NAME_INDEX_PATH = os.path.join("ml_cache", "name_index.json")


def build_name_index(
    all_company_names: List[str],
) -> tuple:
    """
    Pre-normalise all raw OSHA names and return a tuple
    ``(norm_to_raws, choices)`` that can be passed to
    ``group_establishments`` as its *all_company_names* argument.

    ``norm_to_raws``  – dict mapping each normalised string to its raw variants.
    ``choices``       – deduplicated list of normalised strings (for rapidfuzz).
    """
    norm_to_raws: Dict[str, List[str]] = {}
    seen: set = set()
    choices: List[str] = []
    for name in all_company_names:
        n = normalize_establishment_name(name)
        if len(n.strip()) < 2:
            continue
        norm_to_raws.setdefault(n, []).append(name)
        if n not in seen:
            seen.add(n)
            choices.append(n)
    return norm_to_raws, choices


def save_name_index(index: tuple) -> None:
    """Persist the name index to disk as JSON."""
    norm_to_raws, choices = index
    os.makedirs(os.path.dirname(_NAME_INDEX_PATH), exist_ok=True)
    with open(_NAME_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"norm_to_raws": norm_to_raws, "choices": choices}, f)


def load_name_index() -> Optional[tuple]:
    """Load a previously saved name index from disk. Returns None if not found."""
    if not os.path.exists(_NAME_INDEX_PATH):
        return None
    with open(_NAME_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["norm_to_raws"], data["choices"]


def get_or_build_name_index(all_company_names: List[str]) -> tuple:
    """
    Return the cached name index from disk if it exists and matches the
    current name count; otherwise build, save, and return a fresh one.
    """
    cached = load_name_index()
    if cached is not None:
        # Quick count check — compare total raw names stored in the index
        # against the input count, without re-normalizing anything.
        total_raws = sum(len(v) for v in cached[0].values())
        if total_raws == len(all_company_names):
            return cached
    index = build_name_index(all_company_names)
    save_name_index(index)
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

    # ── Step 1: Collect candidates (rapidfuzz WRatio) ────────────────────
    # Use the pre-built normalised index if provided, otherwise build on the fly.
    if isinstance(all_company_names, tuple) and len(all_company_names) == 2:
        norm_to_raws, choices = all_company_names
    else:
        norm_to_raws: Dict[str, List[str]] = {}
        choices: List[str] = []
        for name in all_company_names:
            n = normalize_establishment_name(name)
            if len(n.strip()) < 2:
                continue
            norm_to_raws.setdefault(n, []).append(name)
            if n not in choices:
                choices.append(n)

    if not choices:
        return SearchResultSet(query=query, top_group=None, other_groups=[], unmatched=[])

    # score_cutoff=55 keeps strong variants, rejects noise
    hits = process.extract(
        query_norm, choices,
        scorer=fuzz.WRatio,
        processor=rfutils.default_process,
        score_cutoff=55,
        limit=max_results * 3,
    )

    candidates: List[Tuple[str, float]] = []
    qlen = len(query_norm)
    for norm_name, rf_score, _idx in hits:
        # Reject short candidates that only match via partial_ratio
        if len(norm_name) < qlen * 0.5:
            continue
        score = rf_score / 100.0
        for raw in norm_to_raws[norm_name]:
            candidates.append((raw, score))

    if not candidates:
        return SearchResultSet(query=query, top_group=None, other_groups=[], unmatched=[])

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

    # ── Step 2b: Merge sub-prefix clusters into the primary query cluster ─
    # e.g. query="WALMART": clusters keyed "WALMART SUPERCENTER",
    # "WALMART DISTRIBUTION CENTER", etc. should all fold into "WALMART" so
    # the user sees one group, not 20.  We only merge when a shorter cluster
    # key is a proper word-boundary prefix of a longer one, preventing false
    # merges between unrelated companies that merely share a first word
    # (e.g. "PARKER HANNIFIN" vs "PARKER BROTHERS").
    sorted_keys = sorted(clusters.keys(), key=len)  # shortest first
    merged_clusters: Dict[str, List[Tuple[str, float]]] = {}
    for key in sorted_keys:
        parent_key = None
        for mk in merged_clusters:
            if key.startswith(mk + " ") or key == mk:
                parent_key = mk
                break
        if parent_key is not None:
            merged_clusters[parent_key].extend(clusters[key])
        else:
            merged_clusters[key] = list(clusters[key])
    clusters = merged_clusters

    # ── Step 3: Build GroupedCompanyResult for each cluster ───────────────
    groups: List[GroupedCompanyResult] = []
    unmatched: List[str] = []

    for prefix, members in clusters.items():
        # Determine parent label: use the cluster prefix (already normalised)
        parent_name = prefix.title()

        high  = [(r, s) for r, s in members if s >= 0.85]
        med   = [(r, s) for r, s in members if 0.65 <= s < 0.85]
        low   = [(r, s) for r, s in members if s < 0.65]

        # A cluster with only 1 low-confidence member goes to unmatched
        if len(members) == 1 and members[0][1] < 0.65:
            unmatched.append(members[0][0])
            continue

        # Overall group confidence = mean of top-5 member scores
        top_scores = sorted([s for _, s in members], reverse=True)[:5]
        overall_conf = sum(top_scores) / len(top_scores)

        def _build_facility(raw: str, score: float) -> FacilityCandidate:
            city, state, address, naics_code = _estab_info_for_name(raw, osha_client)
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
                naics_code=naics_code,
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
