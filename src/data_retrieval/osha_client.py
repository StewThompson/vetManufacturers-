import os
import csv
import json
import re
import time
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
from collections import defaultdict
from dotenv import load_dotenv

from src.models.manufacturer import Manufacturer
from src.models.osha_record import OSHARecord, Violation, AccidentSummary

load_dotenv()


class OSHAClient:
    """
    Retrieves OSHA enforcement data using the Department of Labor API (V4).
    Docs: https://data.dol.gov/user-guide
    Endpoints used:
        - OSHA/inspection  (inspection case details)
        - OSHA/violation   (violation + penalty details per inspection)

    Efficiency strategy:
        Heavy fetching is done offline by build_cache.py (run overnight).
        This client loads the pre-built CSV cache from ml_cache/ and
        makes at most 2 lightweight delta API calls for recent records.
    """

    BASE_URL = "https://apiprod.dol.gov/v4/get/OSHA"

    # Cache settings
    CACHE_DIR = "ml_cache"
    INSP_CACHE = "inspections_bulk.csv"
    VIOL_CACHE = "violations_bulk.csv"
    ACC_CACHE = "accidents_bulk.csv"
    INJ_CACHE = "accident_injuries_bulk.csv"
    ABS_CACHE = "accident_abstracts_bulk.csv"
    GEN_DUTY_CACHE = "gen_duty_narratives_bulk.csv"
    CACHE_META = "bulk_meta.json"

    # Accident injury code lookups
    DEGREE_MAP = {"1": "Fatality", "1.0": "Fatality",
                  "2": "Hospitalized", "2.0": "Hospitalized",
                  "3": "Non-hospitalized", "3.0": "Non-hospitalized"}
    NATURE_MAP = {
        "1": "Amputation", "1.0": "Amputation", "2": "Asphyxia", "2.0": "Asphyxia",
        "3": "Bruise/Contusion", "3.0": "Bruise/Contusion",
        "4": "Burn (Chemical)", "4.0": "Burn (Chemical)",
        "5": "Burn/Scald (Heat)", "5.0": "Burn/Scald (Heat)",
        "6": "Concussion", "6.0": "Concussion",
        "7": "Cut/Laceration", "7.0": "Cut/Laceration",
        "8": "Dermatitis", "8.0": "Dermatitis",
        "9": "Dislocation", "9.0": "Dislocation",
        "10": "Electric Shock", "10.0": "Electric Shock",
        "11": "Foreign Body in Eye", "11.0": "Foreign Body in Eye",
        "12": "Fracture", "12.0": "Fracture",
        "13": "Freezing/Frostbite", "13.0": "Freezing/Frostbite",
        "14": "Hearing Loss", "14.0": "Hearing Loss",
        "15": "Heat Exhaustion", "15.0": "Heat Exhaustion",
        "16": "Hernia", "16.0": "Hernia",
        "17": "Poisoning", "17.0": "Poisoning",
        "18": "Puncture", "18.0": "Puncture",
        "19": "Radiation Effects", "19.0": "Radiation Effects",
        "20": "Strain/Sprain", "20.0": "Strain/Sprain",
        "21": "Other", "21.0": "Other", "22": "Cancer", "22.0": "Cancer",
    }
    BODY_MAP = {
        "1": "Abdomen", "1.0": "Abdomen", "2": "Arm (multiple)", "2.0": "Arm (multiple)",
        "3": "Back", "3.0": "Back", "4": "Body System", "4.0": "Body System",
        "5": "Chest", "5.0": "Chest", "6": "Ear(s)", "6.0": "Ear(s)",
        "7": "Elbow(s)", "7.0": "Elbow(s)", "8": "Eye(s)", "8.0": "Eye(s)",
        "9": "Face", "9.0": "Face", "10": "Finger(s)", "10.0": "Finger(s)",
        "11": "Foot/Ankle", "11.0": "Foot/Ankle", "12": "Hand(s)", "12.0": "Hand(s)",
        "13": "Head", "13.0": "Head", "14": "Hip(s)", "14.0": "Hip(s)",
        "15": "Knee(s)", "15.0": "Knee(s)", "16": "Legs", "16.0": "Legs",
        "17": "Lower Arm", "17.0": "Lower Arm", "18": "Lower Leg", "18.0": "Lower Leg",
        "19": "Multiple", "19.0": "Multiple", "20": "Neck", "20.0": "Neck",
        "21": "Shoulder", "21.0": "Shoulder", "22": "Upper Arm", "22.0": "Upper Arm",
        "23": "Upper Leg", "23.0": "Upper Leg", "24": "Wrist(s)", "24.0": "Wrist(s)",
    }

    def __init__(self):
        self.api_key = os.getenv("DOL_API_KEY", "")
        if not self.api_key:
            print("Warning: DOL_API_KEY not set. OSHA API calls will fail.")
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        # In-memory indexes built from cache
        self._inspections_by_estab: Dict[str, list] = defaultdict(list)
        self._violations_by_activity: Dict[str, list] = defaultdict(list)
        self._accidents_by_summary: Dict[str, dict] = {}
        self._injuries_by_inspection: Dict[str, list] = defaultdict(list)
        self._summaries_by_inspection: Dict[str, set] = defaultdict(set)
        self._gen_duty_narratives: Dict[str, str] = {}  # key = "activity_nr|citation_id"

        # Pre-built lists for the search UI
        self._company_names: List[str] = []                    # sorted, title-cased, branch-deduplicated
        self._locations_by_company: Dict[str, List[str]] = {}   # COMPANY_UPPER -> ["123 Main St, City, ST 12345", …]
        self._estab_names_for_company: Dict[str, List[str]] = {}  # COMPANY_UPPER -> [raw ESTAB_UPPER, …]
        self._cache_loaded = False

    # ---- Name-normalization patterns (class-level, compiled once) ----
    # Branch/store-number prefix: "105891 - ", "WA317974334 - ", etc.
    _BRANCH_PREFIX = re.compile(r'^(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,}\s+-\s+')
    # Trailing store/location identifiers
    _SUFFIX_PATTERNS = [
        re.compile(r'\s*[-\u2013]?\s*(?:STORE|DC|PDC|FC|IDC|UNIT|LOC|WAREHOUSE)\s*#?\s*\d+\s*$', re.I),
        re.compile(r'\s*#\s*\d+\s*$'),
        re.compile(r'\s+NO\.?\s+\d+\s*$', re.I),
        re.compile(r'\s+\d{4,}\s*$'),          # bare trailing 4+ digit number
        re.compile(r'\s*[-\u2013]\s*$'),        # dangling hyphen
        re.compile(r'\s*,\s*$'),                # dangling comma
    ]
    # Corporate suffixes (INC, LLC, LP, etc.)
    _CORP_SUFFIX = re.compile(
        r'[,.]?\s*\b(?:INC\.?|LLC\.?|L\.?P\.?|CORP\.?|CO\.?|LTD\.?|'
        r'LIMITED PARTNERSHIP|INCORPORATED|CORPORATION|COMPANY)\s*$', re.I,
    )
    # DBA clauses
    _DBA_CLAUSE = re.compile(r'\s+D/?B/?A\s+.+$', re.I)
    # Trailing/leading English articles ("the", "a", "an") that OSHA sometimes appends
    _TRAILING_THE = re.compile(r'[,.]?\s*\bTHE\s*$', re.I)
    _LEADING_THE  = re.compile(r'^\s*THE\s+', re.I)
    # Normalize spelled-out "AND" / "&" so variants collapse to one key
    _AMP_TO_AND   = re.compile(r'\s*&\s*')
    _AND_SPACES   = re.compile(r'\s+AND\s+', re.I)
    # Collapse runs of whitespace
    _WS           = re.compile(r'\s{2,}')

    @classmethod
    def _normalize_company_name(cls, raw: str) -> str:
        """
        Strip branch prefix, store numbers, corporate suffixes, trailing
        articles, and canonicalize '&' ↔ 'AND' so that equivalent OSHA
        establishment name variants collapse to the same lookup key.

        Examples that all produce "GOODYEAR TIRE AND RUBBER":
          GOODYEAR TIRE & RUBBER
          GOODYEAR TIRE AND RUBBER
          GOODYEAR TIRE AND RUBBER COMPANY
          GOODYEAR TIRE & RUBBER CO THE
          GOODYEAR TIRE & RUBBER, THE
        """
        s = cls._BRANCH_PREFIX.sub('', raw).strip()

        # Iteratively strip store-number suffixes
        changed = True
        while changed:
            changed = False
            for pat in cls._SUFFIX_PATTERNS:
                new = pat.sub('', s).strip()
                if new != s:
                    s = new
                    changed = True

        # Strip DBA clause
        s = cls._DBA_CLAUSE.sub('', s).strip()

        # Strip trailing "THE" before corp-suffix pass (handles "CO THE", ", THE")
        s = cls._TRAILING_THE.sub('', s).strip()

        # Strip corporate suffixes (up to 4 passes for nested cases like "CO, INC.")
        for _ in range(4):
            new = cls._CORP_SUFFIX.sub('', s).strip()
            if new == s:
                break
            s = new

        # Strip trailing "THE" again in case it was hidden behind a corp suffix
        s = cls._TRAILING_THE.sub('', s).strip()
        # Strip leading "THE "
        s = cls._LEADING_THE.sub('', s).strip()

        s = s.rstrip('.,;').strip()

        # Canonicalize ampersand: "& " → " AND " so "Tire & Rubber" == "Tire And Rubber"
        s = cls._AMP_TO_AND.sub(' AND ', s)
        s = cls._AND_SPACES.sub(' AND ', s)

        # Collapse any double-spaces introduced above
        s = cls._WS.sub(' ', s).strip()

        return s

    # ================================================================== #
    #  Cache loading + lightweight delta
    # ================================================================== #
    def ensure_cache(self, force: bool = False) -> bool:
        """
        Load the pre-built bulk cache from disk.
        The heavy fetching is done offline by build_cache.py.
        This method only loads the cache into memory and optionally
        fetches a small delta (records newer than cache date).
        """
        if self._cache_loaded and not force:
            return True

        self._load_cache_from_disk()
        if not self._cache_loaded:
            print("No OSHA cache found. Run build_cache.py first.")
            return False

        # Lightweight delta: fetch only records newer than the cache date
        meta_path = os.path.join(self.CACHE_DIR, self.CACHE_META)
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                cache_date = meta.get("date")
                if cache_date:
                    self._delta_fetch(cache_date)
            except Exception as e:
                print(f"Delta fetch skipped: {e}")

        return True

    def _delta_fetch(self, since_date: str):
        """
        Fetch only records newer than since_date and merge into memory.
        At most 2 lightweight API calls (inspections + violations).
        """
        print(f"  Delta fetch for records after {since_date}…")

        insp_filter = json.dumps(
            {"field": "open_date", "operator": "gt", "value": since_date}
        )
        new_insp = self._api_get("inspection", {
            "limit": "500", "sort": "desc", "sort_by": "open_date",
            "fields": "activity_nr,estab_name,site_address,site_city,"
                      "site_state,site_zip,open_date,close_case_date,"
                      "insp_type,insp_scope",
            "filter_object": insp_filter,
        }, timeout=15)
        for insp in new_insp:
            name = (insp.get("estab_name") or "").upper()
            self._inspections_by_estab[name].append(insp)

        viol_filter = json.dumps({"and": [
            {"field": "current_penalty", "operator": "gt", "value": "0"},
            {"field": "issuance_date", "operator": "gt", "value": since_date},
        ]})
        new_viol = self._api_get("violation", {
            "limit": "500", "sort": "desc", "sort_by": "issuance_date",
            "fields": "activity_nr,citation_id,standard,viol_type,"
                      "current_penalty,initial_penalty,issuance_date,"
                      "abate_date,delete_flag",
            "filter_object": viol_filter,
        }, timeout=15)
        for v in new_viol:
            if v.get("delete_flag") == "X":
                continue
            act = str(v.get("activity_nr", ""))
            self._violations_by_activity[act].append(v)

        if new_insp or new_viol:
            print(f"  Delta: +{len(new_insp)} inspections, +{len(new_viol)} violations.")
        else:
            print("  Delta: no new records.")

    @staticmethod
    def _read_csv_cache(path: str) -> list:
        """Read a cached CSV file into a list of dicts."""
        if not os.path.exists(path):
            return []
        # Gen-duty narratives can exceed the default 128 KB field limit;
        # raise it to the system max so long inspector notes don't crash the reader.
        csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _load_cache_from_disk(self):
        """Load cached CSV files into memory indexes (falls back to legacy JSON)."""
        insp_path = os.path.join(self.CACHE_DIR, self.INSP_CACHE)
        viol_path = os.path.join(self.CACHE_DIR, self.VIOL_CACHE)
        inspections = self._read_csv_cache(insp_path)
        if not inspections:
            json_path = os.path.join(self.CACHE_DIR, "inspections_bulk.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        inspections = json.load(f)
                except Exception:
                    pass
        if not inspections:
            print("No inspection cache found.")
            return
        violations = self._read_csv_cache(viol_path)
        if not violations:
            json_path = os.path.join(self.CACHE_DIR, "violations_bulk.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        violations = json.load(f)
                except Exception:
                    violations = []

        # Accident + injury data (optional — won't block loading)
        acc_path = os.path.join(self.CACHE_DIR, self.ACC_CACHE)
        inj_path = os.path.join(self.CACHE_DIR, self.INJ_CACHE)
        accidents = self._read_csv_cache(acc_path)
        injuries = self._read_csv_cache(inj_path)

        # Gen-duty narratives (optional — not present until build_cache.py has been run)
        gd_path = os.path.join(self.CACHE_DIR, self.GEN_DUTY_CACHE)
        gd_rows = self._read_csv_cache(gd_path)
        self._gen_duty_narratives = {}
        for row in gd_rows:
            act = row.get("activity_nr", "")
            cit = row.get("citation_id", "")
            if act and cit:
                self._gen_duty_narratives[f"{act}|{cit}"] = row.get("narrative_text", "")

        self._build_indexes(inspections, violations, accidents, injuries)
        parts = [f"{len(inspections)} inspections", f"{len(violations)} violations"]
        if accidents:
            parts.append(f"{len(accidents)} accidents")
        if injuries:
            parts.append(f"{len(injuries)} injuries")
        if gd_rows:
            parts.append(f"{len(self._gen_duty_narratives)} gen duty narratives")
        print(f"Loaded bulk cache: {', '.join(parts)}.")

    def _build_indexes(self, inspections: list, violations: list,
                       accidents: list = None, injuries: list = None):
        """Build in-memory lookup dicts from raw data lists."""
        self._inspections_by_estab = defaultdict(list)
        self._violations_by_activity = defaultdict(list)

        for insp in inspections:
            name = (insp.get("estab_name") or "").upper()
            self._inspections_by_estab[name].append(insp)

        for v in violations:
            if v.get("delete_flag") == "X":
                continue
            act = str(v.get("activity_nr", ""))
            self._violations_by_activity[act].append(v)

        # Accident indexes
        self._accidents_by_summary = {}
        self._injuries_by_inspection = defaultdict(list)
        self._summaries_by_inspection = defaultdict(set)

        for acc in (accidents or []):
            snr = str(acc.get("summary_nr", ""))
            if snr:
                self._accidents_by_summary[snr] = acc

        for inj in (injuries or []):
            rel = str(inj.get("rel_insp_nr", ""))
            snr = str(inj.get("summary_nr", ""))
            if rel:
                self._injuries_by_inspection[rel].append(inj)
                if snr:
                    self._summaries_by_inspection[rel].add(snr)

        # Build company-level indexes (normalized & deduplicated)
        company_estabs: Dict[str, list] = defaultdict(list)  # COMPANY_UPPER -> [ESTAB_UPPER, ...]
        for estab_upper in self._inspections_by_estab:
            if not estab_upper:
                continue
            clean = self._normalize_company_name(estab_upper).upper()
            if clean:
                company_estabs[clean].append(estab_upper)

        self._company_names = sorted(
            {name.title() for name in company_estabs},
            key=str.casefold,
        )
        self._estab_names_for_company = {k: v for k, v in company_estabs.items()}

        # Full-address locations per company
        self._locations_by_company = {}
        for company_upper, estab_list in company_estabs.items():
            seen_locs: set = set()
            locs: List[str] = []
            for estab_upper in estab_list:
                for insp in self._inspections_by_estab.get(estab_upper, []):
                    addr = (insp.get("site_address") or "").strip()
                    city = (insp.get("site_city") or "").strip()
                    state = (insp.get("site_state") or "").strip()
                    zip_code = (insp.get("site_zip") or "").strip()
                    parts = [p for p in (addr, city, state + (" " + zip_code if zip_code else "")) if p]
                    loc = ", ".join(parts)
                    if loc and loc not in seen_locs:
                        seen_locs.add(loc)
                        locs.append(loc)
            if locs:
                self._locations_by_company[company_upper] = locs

        self._cache_loaded = True

    # ================================================================== #
    #  Public API
    # ================================================================== #
    def search_manufacturer(self, manufacturer: Manufacturer) -> List[OSHARecord]:
        """
        Look up a manufacturer's OSHA records.
        Searches the bulk cache first (0 API calls).
        Falls back to a targeted API call only if cache is empty.
        """
        print(f"Searching OSHA records for: {manufacturer.name}")

        # Try bulk cache first
        self.ensure_cache()

        # location may be a single string or None; _search_cache expects a list
        loc_list = [manufacturer.location] if manufacturer.location else None
        records = self._search_cache(manufacturer.name, loc_list)
        if records is not None:
            return records

        # Fallback: targeted API call for this manufacturer only (2 calls max)
        print("  Manufacturer not in bulk cache — making targeted API call…")
        return self._search_api(manufacturer.name)

    def get_bulk_inspections(self) -> list:
        """Return all cached raw inspection dicts (for ML scorer)."""
        self.ensure_cache()
        # Flatten the index
        all_insp = []
        for insp_list in self._inspections_by_estab.values():
            all_insp.extend(insp_list)
        return all_insp

    def get_bulk_violations(self) -> list:
        """Return all cached raw violation dicts (for ML scorer)."""
        self.ensure_cache()
        all_viol = []
        for viol_list in self._violations_by_activity.values():
            all_viol.extend(viol_list)
        return all_viol

    def get_violations_for_activity(self, activity_nr: str) -> list:
        """Look up violations by activity_nr from the cache."""
        self.ensure_cache()
        return self._violations_by_activity.get(str(activity_nr), [])

    def get_injuries_for_inspection(self, activity_nr: str) -> list:
        """Return decoded injury records linked to an inspection."""
        self.ensure_cache()
        raw = self._injuries_by_inspection.get(str(activity_nr), [])
        return [self._decode_injury(inj) for inj in raw]

    def get_accidents_for_inspection(self, activity_nr: str) -> List[AccidentSummary]:
        """Return AccidentSummary objects linked to an inspection."""
        self.ensure_cache()
        summaries = self._summaries_by_inspection.get(str(activity_nr), set())
        results = []
        for snr in summaries:
            acc = self._accidents_by_summary.get(snr, {})
            injuries = [self._decode_injury(inj)
                        for inj in self._injuries_by_inspection.get(str(activity_nr), [])
                        if str(inj.get("summary_nr", "")) == snr]
            fatality_flag = str(acc.get("fatality", "")).strip()
            is_fatality = fatality_flag in ("1", "Y", "True")
            # Also check if any linked injury has degree_of_inj == 1 (Fatality)
            if not is_fatality:
                is_fatality = any(inj.get("degree") == "Fatality" for inj in injuries)
            results.append(AccidentSummary(
                summary_nr=snr,
                event_date=acc.get("event_date", "")[:10] if acc.get("event_date") else None,
                event_desc=acc.get("event_desc", ""),
                fatality=is_fatality,
                injuries=injuries,
            ))
        return results

    # ---- General Duty helpers ----
    _GEN_DUTY_RE = re.compile(r'^5A1|^OSHACT$|^SECTION5$|^GENERALDUTY', re.I)

    @classmethod
    def _is_gen_duty_standard(cls, standard: str) -> bool:
        """True if the standard code is a General Duty Clause (Section 5(a)(1)) citation."""
        clean = re.sub(r"[\s.\-/()']+", "", (standard or "").upper())
        return bool(cls._GEN_DUTY_RE.match(clean))

    @staticmethod
    def _is_high_priority_violation(v: Violation) -> bool:
        """True for violations warranting a detailed narrative: willful, repeat, high gravity, or significant penalty."""
        if v.is_willful or v.is_repeat:
            return True
        if v.gravity:
            try:
                if int(v.gravity) >= 8:
                    return True
            except (ValueError, TypeError):
                pass
        if v.penalty_amount > 5000:
            return True
        return False

    def get_gen_duty_narrative(self, activity_nr: str, citation_id: str) -> str:
        """Return the inspector narrative for a specific Gen Duty citation (on-demand lookup)."""
        self.ensure_cache()
        return self._gen_duty_narratives.get(f"{activity_nr}|{citation_id}", "")

    def get_accident_abstract(self, summary_nr: str) -> str:
        """Load the pre-joined abstract text for a specific accident (disk read)."""
        abs_path = os.path.join(self.CACHE_DIR, self.ABS_CACHE)
        if not os.path.exists(abs_path):
            return ""
        with open(abs_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("summary_nr") == str(summary_nr):
                    return row.get("abstract_text", "")
        return ""

    def _decode_injury(self, inj: dict) -> dict:
        """Decode coded injury fields into human-readable labels."""
        return {
            "degree": self.DEGREE_MAP.get(str(inj.get("degree_of_inj", "")), "Unknown"),
            "nature": self.NATURE_MAP.get(str(inj.get("nature_of_inj", "")), "Unknown"),
            "body_part": self.BODY_MAP.get(str(inj.get("part_of_body", "")), "Unknown"),
            "age": inj.get("age", ""),
            "sex": inj.get("sex", ""),
        }

    def get_accident_count_for_activity(self, activity_nr: str) -> dict:
        """Quick stats: accident count, fatalities, injuries for one inspection."""
        self.ensure_cache()
        summaries = self._summaries_by_inspection.get(str(activity_nr), set())
        fatalities = 0
        injury_count = 0
        for snr in summaries:
            acc = self._accidents_by_summary.get(snr, {})
            is_fatal = str(acc.get("fatality", "")).strip() in ("1", "Y", "True")
            injs_for_snr = [inj for inj in self._injuries_by_inspection.get(str(activity_nr), [])
                            if str(inj.get("summary_nr", "")) == snr]
            # Also check injury degree for fatality
            if not is_fatal:
                is_fatal = any(str(inj.get("degree_of_inj", "")).startswith("1") for inj in injs_for_snr)
            if is_fatal:
                fatalities += 1
            injury_count += len(injs_for_snr)
        return {"accidents": len(summaries), "fatalities": fatalities, "injuries": injury_count}

    # ================================================================== #
    #  Cache search
    # ================================================================== #
    def get_all_company_names(self) -> List[str]:
        """Return sorted, branch-deduplicated company names (title-cased)."""
        self.ensure_cache()
        return self._company_names

    def get_locations_for_company(self, company: str) -> List[str]:
        """Return full-address locations recorded for *company*."""
        self.ensure_cache()
        return self._locations_by_company.get(company.strip().upper(), [])

    def _search_cache(self, name: str, locations: Optional[List[str]] = None) -> Optional[List[OSHARecord]]:
        """
        Find matching inspections in the cache by company name.
        *name* is a branch-deduplicated company name; we look up all raw
        estab keys that belong to it, then optionally filter by *locations*
        (list of full-address strings the user selected).
        Returns None if no match found (caller should fall back to API).
        """
        search = name.strip().upper()

        # Resolve company -> underlying estab keys
        estab_keys = self._estab_names_for_company.get(search)
        if not estab_keys:
            # Fallback: try raw estab index directly
            if search in self._inspections_by_estab:
                estab_keys = [search]
            else:
                # Substring search
                estab_keys = [e for e in self._inspections_by_estab if search in e or e in search]

        if not estab_keys:
            return None

        matches = []
        for ek in estab_keys:
            matches.extend(self._inspections_by_estab.get(ek, []))

        if not matches:
            return None

        # Filter by selected locations (full-address match)
        if locations:
            loc_set = {loc.upper() for loc in locations}
            filtered = []
            for m in matches:
                addr = (m.get("site_address") or "").strip()
                city = (m.get("site_city") or "").strip()
                state = (m.get("site_state") or "").strip()
                zip_code = (m.get("site_zip") or "").strip()
                parts = [p for p in (addr, city, state + (" " + zip_code if zip_code else "")) if p]
                full = ", ".join(parts).upper()
                if full in loc_set:
                    filtered.append(m)
            if filtered:
                matches = filtered

        print(f"  Found {len(matches)} inspection(s) in cache.")
        return self._build_records(matches)

    def _build_records(self, inspections: list) -> List[OSHARecord]:
        """Convert raw inspection dicts + cached violations + accidents into OSHARecord objects."""
        records: List[OSHARecord] = []
        for insp in inspections:
            activity_nr = str(insp.get("activity_nr", ""))
            date_str = insp.get("open_date", "")
            try:
                date_opened = (datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                               if date_str else date.today())
            except ValueError:
                date_opened = date.today()

            raw_viols = self._violations_by_activity.get(activity_nr, [])
            violations, total_penalties = self._parse_violations(raw_viols)

            # Attach inspector narratives for high-priority General Duty violations only
            if self._gen_duty_narratives:
                for v in violations:
                    if self._is_gen_duty_standard(v.category) and self._is_high_priority_violation(v):
                        narrative = self._gen_duty_narratives.get(f"{activity_nr}|{v.citation_id or ''}", "")
                        if narrative:
                            v.gen_duty_narrative = narrative

            # Accident data — linked via injury records
            accidents = self.get_accidents_for_inspection(activity_nr)
            # Severe only if there are actual linked accidents or
            # the inspection was a Fat/Cat type ("L")
            severe = bool(accidents) or insp.get("insp_type", "") == "L"

            records.append(OSHARecord(
                inspection_id=activity_nr,
                date_opened=date_opened,
                violations=violations,
                total_penalties=total_penalties,
                severe_injury_or_fatality=severe,
                accidents=accidents,
                naics_code=insp.get("naics_code"),
                nr_in_estab=insp.get("nr_in_estab"),
            ))
        return records

    # ================================================================== #
    #  Targeted API fallback (2 calls: inspections + violations)
    # ================================================================== #
    def _search_api(self, name: str) -> List[OSHARecord]:
        """
        Targeted search when a manufacturer isn't in the bulk cache.
        Makes exactly 2 API calls: one for inspections, one for all
        their violations (using activity_nr list).
        """
        inspections = self._get_inspections(name)
        if not inspections:
            print("  No OSHA inspections found via API.")
            return []

        # Collect all activity_nrs, then fetch violations in one batch
        activity_nrs = [str(i.get("activity_nr", "")) for i in inspections]
        self._batch_fetch_violations(activity_nrs)

        return self._build_records(inspections)

    def _batch_fetch_violations(self, activity_nrs: list):
        """
        Fetch violations for a list of activity_nrs in ONE API call
        using a compound 'or' filter with multiple 'eq' conditions.
        Checks the in-memory cache first; only queries missing ones.
        """
        if not activity_nrs:
            return

        # Filter to only activity_nrs not already cached
        missing = [a for a in activity_nrs
                   if a and a not in self._violations_by_activity]

        if not missing:
            return

        print(f"  Fetching violations for {len(missing)} inspection(s) via API (1 call)…")

        # Build compound OR filter: {"or": [{"field":..,"operator":"eq","value":nr}, ...]}
        conditions = [
            {"field": "activity_nr", "operator": "eq", "value": nr}
            for nr in missing
        ]
        if len(conditions) == 1:
            filter_obj = json.dumps(conditions[0])
        else:
            filter_obj = json.dumps({"or": conditions})

        params = {
            "limit": "500",
            "fields": "activity_nr,citation_id,standard,viol_type,"
                      "current_penalty,initial_penalty,issuance_date,"
                      "abate_date,delete_flag",
            "filter_object": filter_obj,
        }
        data = self._api_get("violation", params)
        for v in data:
            if v.get("delete_flag") == "X":
                continue
            act = str(v.get("activity_nr", ""))
            self._violations_by_activity[act].append(v)

    # ================================================================== #
    #  Low-level helpers
    # ================================================================== #
    MAX_RETRIES = 3
    RETRY_BACKOFF = [5, 15, 30]  # seconds

    def _api_get(self, endpoint: str, params: dict, timeout: int = 60) -> list:
        """Makes a GET request to the DOL API with retry on 429 rate-limit."""
        url = f"{self.BASE_URL}/{endpoint}/json"
        params["X-API-KEY"] = self.api_key

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                if resp.status_code == 429:
                    wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                    print(f"  Rate-limited (429). Waiting {wait}s before retry {attempt + 1}…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                if resp.status_code == 204 or not resp.text.strip():
                    return []
                data = resp.json()
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data if isinstance(data, list) else []
            except requests.exceptions.HTTPError as e:
                print(f"DOL API HTTP error: {e} – {resp.text[:300]}")
                return []
            except requests.exceptions.RequestException as e:
                print(f"DOL API request error: {e}")
                return []
            except (json.JSONDecodeError, ValueError):
                print("DOL API returned non-JSON response.")
                return []

        print("DOL API: max retries exhausted.")
        return []

    def _get_inspections(self, name: str) -> list:
        """Query the OSHA/inspection endpoint, filtering by estab_name LIKE %name%."""
        search_name = name.strip().upper()
        filter_obj = json.dumps({
            "field": "estab_name",
            "operator": "like",
            "value": f"%{search_name}%"
        })
        params = {
            "limit": "250",
            "sort": "desc",
            "sort_by": "open_date",
            "fields": "activity_nr,estab_name,site_address,site_city,site_state,"
                      "site_zip,open_date,close_case_date,insp_type,insp_scope",
            "filter_object": filter_obj,
        }
        return self._api_get("inspection", params)

    @staticmethod
    def _parse_violations(raw_viols: list) -> Tuple[List[Violation], float]:
        """Convert raw violation dicts into Violation objects + total penalty."""
        violations: List[Violation] = []
        total_penalties = 0.0
        severity_map = {"S": "Serious", "W": "Willful", "R": "Repeat", "O": "Other"}

        for row in raw_viols:
            viol_type = row.get("viol_type", "O")
            penalty = float(row.get("current_penalty")
                            or row.get("initial_penalty") or 0)
            standard = row.get("standard", "Unknown")
            severity = severity_map.get(viol_type, "Other")
            gravity = row.get("gravity", "")
            try:
                nr_exposed = float(row.get("nr_exposed") or 0)
            except (ValueError, TypeError):
                nr_exposed = None

            # Hazardous substances
            hazsubs = [row.get(f"hazsub{i}", "") for i in range(1, 6)]
            hazsub_str = ", ".join(h for h in hazsubs if h and h.strip()) or None

            violations.append(Violation(
                category=standard,
                severity=severity,
                penalty_amount=penalty,
                is_repeat=(viol_type == "R"),
                is_willful=(viol_type == "W"),
                description=f"{standard} ({severity})",
                gravity=gravity if gravity else None,
                nr_exposed=nr_exposed,
                hazardous_substance=hazsub_str,
                citation_id=row.get("citation_id", "") or None,
            ))
            total_penalties += penalty

        return violations, total_penalties
