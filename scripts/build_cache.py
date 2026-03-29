"""
Consolidate downloaded OSHA CSV chunks (OshaData/) into ml_cache/ for fast
in-memory lookup.  Also pre-joins accident abstract lines into single texts.

Usage:
    py build_cache.py              # full build from local OshaData/

Datasets produced in ml_cache/:
    inspections_bulk.csv       – 10-year rolling window, key inspection fields
    violations_bulk.csv        – 10-year rolling window, all violations (incl. zero-penalty)
    accidents_bulk.csv         – all accident events
    accident_injuries_bulk.csv – all injury records (links to inspections)
    accident_abstracts_bulk.csv – pre-joined full-text abstracts (one row per accident)
"""

import os
import csv
import json
import glob
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta

OSHA_DIR = "OshaData"
CACHE_DIR = "ml_cache"
META_FILE = os.path.join(CACHE_DIR, "bulk_meta.json")
# Rolling window for inspections and violations.
# 3 years was too short — it missed violations from 2014-2020 that are still
# predictive of current compliance behaviour.
# Full OSHA history (1972-present) is too long — reformed companies whose
# violations predate 2010 score high but have good recent outcomes, causing
# prediction quality to degrade.
# 10 years captures a meaningful recent history that balances both concerns.
CUTOFF_YEARS = 10


def _natural_sort_key(path):
    """Sort chunk_1, chunk_2, ..., chunk_10, chunk_100 correctly."""
    parts = re.split(r'(\d+)', os.path.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def iter_chunks(folder_name):
    """Yield rows from all CSV chunks in a folder, headers lowercased."""
    folder = os.path.join(OSHA_DIR, folder_name)
    files = sorted(glob.glob(os.path.join(folder, "*.csv")), key=_natural_sort_key)
    if not files:
        print(f"  WARNING: no CSV files in {folder}")
        return
    print(f"  Reading {len(files)} chunk(s) from {folder_name}...")
    for i, fp in enumerate(files, 1):
        if i % 20 == 0 or i == len(files):
            print(f"    chunk {i}/{len(files)}...")
        with open(fp, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield {k.lower(): v for k, v in row.items()}


def build_dataset(folder_name, output_name, date_field=None,
                  cutoff_date=None, fields=None):
    """Read chunks, optionally filter by date, write a single CSV. Returns row count."""
    out_path = os.path.join(CACHE_DIR, output_name)
    count = 0
    writer = None

    with open(out_path, "w", newline="", encoding="utf-8") as outf:
        for row in iter_chunks(folder_name):
            if date_field and cutoff_date:
                val = row.get(date_field, "")
                if not val or val[:10] < cutoff_date:
                    continue

            if fields:
                row = {k: row.get(k, "") for k in fields}

            if writer is None:
                writer = csv.DictWriter(outf, fieldnames=list(row.keys()))
                writer.writeheader()

            writer.writerow(row)
            count += 1

            if count % 200_000 == 0:
                print(f"    {count:,} rows...")

    return count


def build_gen_duty_narratives():
    """
    Read all General Duty Clause narrative lines, group by (activity_nr, citation_id),
    concatenate in line order, and write one row per citation.
    Returns count of unique narratives.
    """
    print("\nBuilding General Duty narratives (pre-joining lines)\u2026")
    narratives: dict[tuple, list] = defaultdict(list)

    for row in iter_chunks("OSHA_violation_gen_duty_std"):
        act = row.get("activity_nr", "")
        cit = row.get("citation_id", "")
        if not act or not cit:
            continue
        try:
            line_nr = int(float(row.get("line_nr", 0)))
        except (ValueError, TypeError):
            line_nr = 0
        text = row.get("line_text", "")
        narratives[(act, cit)].append((line_nr, text))

    out_path = os.path.join(CACHE_DIR, "gen_duty_narratives_bulk.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["activity_nr", "citation_id", "narrative_text"])
        writer.writeheader()
        for (act, cit), lines in narratives.items():
            lines.sort(key=lambda x: x[0])
            full_text = " ".join(t for _, t in lines if t).strip()
            if full_text:
                writer.writerow({"activity_nr": act, "citation_id": cit, "narrative_text": full_text})

    print(f"  \u2192 {len(narratives):,} unique Gen Duty narratives")
    return len(narratives)


def build_abstracts():
    """
    Read all abstract lines, group by summary_nr, concatenate in line
    order, and write one row per accident. Returns count of unique abstracts.
    """
    print("\nBuilding accident abstracts (pre-joining lines)...")
    abstracts: dict[str, list] = defaultdict(list)

    for row in iter_chunks("OSHA_accident_abstract"):
        snr = row.get("summary_nr", "")
        if not snr:
            continue
        try:
            line_nr = int(float(row.get("line_nr", 0)))
        except (ValueError, TypeError):
            line_nr = 0
        text = row.get("abstract_text", "")
        abstracts[snr].append((line_nr, text))

    out_path = os.path.join(CACHE_DIR, "accident_abstracts_bulk.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["summary_nr", "abstract_text"])
        writer.writeheader()
        for snr, lines in abstracts.items():
            lines.sort(key=lambda x: x[0])
            full_text = " ".join(t for _, t in lines if t).strip()
            if full_text:
                writer.writerow({"summary_nr": snr, "abstract_text": full_text})

    print(f"  -> {len(abstracts):,} unique abstracts")
    return len(abstracts)


# ------------------------------------------------------------------ #
#  SQLite cache builder
# ------------------------------------------------------------------ #
def build_sqlite_db():
    """
    Build osha_cache.db from the pre-built CSVs in ml_cache/.
    Called after all CSV/JSON files are written by main().
    """
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    try:
        from src.data_retrieval.osha_client import OSHAClient
        normalize = OSHAClient.company_match_key
    except Exception:
        def normalize(s): return s  # fallback: no normalization

    db_path = os.path.join(CACHE_DIR, "osha_cache.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"\nBuilding SQLite cache \u2192 {os.path.basename(db_path)} \u2026")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache

    def _load_csv_to_table(csv_path, table_name, extra_cols=None,
                           row_transform=None, skip_row=None):
        """Read a CSV into a SQLite table. Returns row count."""
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found \u2014 skipping {table_name}")
            return 0
        csv.field_size_limit(10 * 1024 * 1024)
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames or []
        all_cols = list(fieldnames) + (extra_cols or [])
        col_defs = ", ".join(f'"{c}" TEXT' for c in all_cols)
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')
        placeholders = ", ".join("?" for _ in all_cols)
        insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
        count = 0
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if skip_row and skip_row(row):
                    continue
                base_vals = [row.get(c, "") for c in fieldnames]
                extra_vals = row_transform(row) if row_transform else []
                conn.execute(insert_sql, base_vals + extra_vals)
                count += 1
                if count % 100_000 == 0:
                    conn.commit()
                    print(f"    {count:,} rows\u2026")
        conn.commit()
        return count

    # ---- inspections (+ computed company_key) ----
    company_keys: set = set()

    def _insp_transform(row):
        raw = (row.get("estab_name") or "").upper()
        key = normalize(raw).upper() if raw else ""
        company_keys.add(key)
        return [key]

    insp_count = _load_csv_to_table(
        os.path.join(CACHE_DIR, "inspections_bulk.csv"),
        "inspections",
        extra_cols=["company_key"],
        row_transform=_insp_transform,
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_insp_company_key ON inspections(company_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_insp_estab_name  ON inspections(estab_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_insp_activity_nr ON inspections(activity_nr)")
    conn.commit()
    print(f"  \u2192 {insp_count:,} inspections")

    # ---- company_names ----
    conn.execute("CREATE TABLE company_names (company_key TEXT PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT OR IGNORE INTO company_names VALUES (?, ?)",
        [(k, k.title()) for k in sorted(company_keys) if k],
    )
    conn.commit()
    print(f"  \u2192 {len([k for k in company_keys if k]):,} unique company names")

    # ---- violations (skip deleted rows) ----
    viol_count = _load_csv_to_table(
        os.path.join(CACHE_DIR, "violations_bulk.csv"),
        "violations",
        skip_row=lambda r: r.get("delete_flag") == "X",
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_activity_nr ON violations(activity_nr)")
    conn.commit()
    print(f"  \u2192 {viol_count:,} violations")

    # ---- accidents ----
    acc_count = _load_csv_to_table(
        os.path.join(CACHE_DIR, "accidents_bulk.csv"), "accidents"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_acc_summary_nr ON accidents(summary_nr)")
    conn.commit()
    print(f"  \u2192 {acc_count:,} accidents")

    # ---- injuries ----
    inj_count = _load_csv_to_table(
        os.path.join(CACHE_DIR, "accident_injuries_bulk.csv"), "injuries"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inj_rel_insp_nr ON injuries(rel_insp_nr)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inj_summary_nr  ON injuries(summary_nr)")
    conn.commit()
    print(f"  \u2192 {inj_count:,} injuries")

    # ---- gen_duty_narratives ----
    gd_count = _load_csv_to_table(
        os.path.join(CACHE_DIR, "gen_duty_narratives_bulk.csv"), "gen_duty_narratives"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_gdn_act_cit "
        "ON gen_duty_narratives(activity_nr, citation_id)"
    )
    conn.commit()
    print(f"  \u2192 {gd_count:,} gen duty narratives")

    # ---- accident_abstracts ----
    abs_count = _load_csv_to_table(
        os.path.join(CACHE_DIR, "accident_abstracts_bulk.csv"), "accident_abstracts"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_abs_summary_nr ON accident_abstracts(summary_nr)"
    )
    conn.commit()
    print(f"  \u2192 {abs_count:,} accident abstracts")

    conn.close()
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  SQLite cache ready: {db_path} ({db_size_mb:.1f} MB)")


# ------------------------------------------------------------------ #
#  Multi-target model helpers
# ------------------------------------------------------------------ #
def _build_penalty_percentiles(scorer):
    """Compute NAICS-stratified P75/P90/P95 from violations_bulk.csv.

    Uses only training-period data (pre-TEMPORAL_LABEL_CUTOFF) to avoid
    leakage into the multi-target outcome labels.
    """
    import pandas as pd
    from src.scoring.penalty_percentiles import (
        compute_penalty_percentiles,
        save_percentiles,
        CACHE_FILENAME,
    )

    viol_path = os.path.join(CACHE_DIR, "violations_bulk.csv")
    if not os.path.exists(viol_path):
        print("  WARNING: violations_bulk.csv not found; skipping penalty percentiles.")
        return

    cutoff_str = scorer.TEMPORAL_LABEL_CUTOFF.isoformat()
    insp_path  = os.path.join(CACHE_DIR, "inspections_bulk.csv")

    # Build activity_nr → naics_code mapping from inspections (violations lack naics_code)
    naics_by_act: dict = {}
    if os.path.exists(insp_path):
        print("  Building activity_nr → NAICS map from inspections …")
        with open(insp_path, "r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                act   = row.get("activity_nr", "").strip()
                naics = row.get("naics_code",  "").strip()
                if act and naics:
                    naics_by_act[act] = naics[:2]
        print(f"  {len(naics_by_act):,} inspection records indexed")

    print(f"  Reading violations pre-{cutoff_str} for threshold computation …")

    rows = []
    csv.field_size_limit(10 * 1024 * 1024)
    with open(viol_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            iso = row.get("issuance_date", "")[:10]
            if iso and iso < cutoff_str:
                penalty = float(
                    row.get("current_penalty") or row.get("initial_penalty") or 0
                )
                act      = row.get("activity_nr", "").strip()
                naics_2d = naics_by_act.get(act, "__unknown__")
                if penalty > 0:
                    rows.append({"naics_2digit": naics_2d, "penalty_amount": penalty})

    if not rows:
        print("  No pre-cutoff violations found; skipping penalty percentiles.")
        return

    df = pd.DataFrame(rows)
    thresholds = compute_penalty_percentiles(df, min_group_n=50)
    out_path   = os.path.join(CACHE_DIR, CACHE_FILENAME)
    save_percentiles(thresholds, out_path)
    print(
        f"  Penalty percentiles: {len(thresholds)-1} NAICS groups + global fallback "
        f"→ {out_path}"
    )
    global_t = thresholds["__global__"]
    print(
        f"    Global: P75=${global_t['p75']:,.0f}  "
        f"P90=${global_t['p90']:,.0f}  P95=${global_t['p95']:,.0f}"
    )


def _build_multi_target_model(scorer):
    """Train and save MultiTargetRiskScorer from the bulk cache CSVs."""
    import numpy as np
    from src.scoring.multi_target_labeler import load_or_build as _mt_load_or_build
    from src.scoring.multi_target_scorer import MultiTargetRiskScorer, MODEL_FILE
    from src.scoring.penalty_percentiles import load_percentiles, CACHE_FILENAME
    from datetime import date as _date

    insp_path   = os.path.join(CACHE_DIR, "inspections_bulk.csv")
    viol_path   = os.path.join(CACHE_DIR, "violations_bulk.csv")
    acc_path    = os.path.join(CACHE_DIR, "accidents_bulk.csv")
    inj_path    = os.path.join(CACHE_DIR, "accident_injuries_bulk.csv")
    thresh_path = os.path.join(CACHE_DIR, CACHE_FILENAME)

    for p, label in [
        (insp_path,  "inspections_bulk.csv"),
        (viol_path,  "violations_bulk.csv"),
    ]:
        if not os.path.exists(p):
            print(f"  WARNING: {label} not found; skipping multi-target model.")
            return

    penalty_thresholds = load_percentiles(thresh_path)
    cutoff_date        = scorer.TEMPORAL_LABEL_CUTOFF
    outcome_end        = _date.today()

    rows = _mt_load_or_build(
        scorer=scorer,
        cutoff_date=cutoff_date,
        outcome_end_date=outcome_end,
        cache_dir=CACHE_DIR,
        inspections_path=insp_path,
        violations_path=viol_path,
        accidents_path=acc_path,
        injuries_path=inj_path,
        naics_map=scorer._naics_map,
        penalty_thresholds=penalty_thresholds,
        sample_size=scorer.TEMPORAL_SAMPLE_SIZE,
    )

    if len(rows) < 50:
        print(f"  Only {len(rows)} multi-target rows; skipping model training.")
        return

    # Build feature matrix (log-transformed, same as training)
    X_raw = np.array([r["features_46"] for r in rows], dtype=float)
    X     = scorer._log_transform_features(
        np.nan_to_num(X_raw, nan=0.0)
    )

    mt_scorer = MultiTargetRiskScorer()
    mt_scorer.fit(X, rows, optimize_weights=True, val_fraction=0.20)

    out_path = os.path.join(CACHE_DIR, MODEL_FILE)
    mt_scorer.save(out_path)
    print(f"  Multi-target model saved → {out_path}  ({len(rows):,} training rows)")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cutoff = (date.today() - timedelta(days=CUTOFF_YEARS * 365)).isoformat()

    print("OSHA Cache Builder (local data)")
    print(f"  Source: {OSHA_DIR}/")
    print(f"  Target: {CACHE_DIR}/")
    print(f"  Date cutoff (rolling {CUTOFF_YEARS}-year window): {cutoff}")

    # --- Inspections (10-year rolling window) ---
    print(f"\nBuilding inspections (open_date >= {cutoff})...")
    insp_fields = [
        "activity_nr", "estab_name", "site_address", "site_city",
        "site_state", "site_zip", "owner_type", "naics_code",
        "insp_type", "insp_scope", "safety_hlth", "open_date",
        "close_case_date", "close_conf_date", "nr_in_estab",
    ]
    insp_count = build_dataset(
        "OSHA_inspection", "inspections_bulk.csv",
        date_field="open_date", cutoff_date=cutoff, fields=insp_fields,
    )
    print(f"  -> {insp_count:,} inspections")

    # --- Violations (10-year rolling window, all penalties) ---
    print(f"\nBuilding violations (issuance_date >= {cutoff})...")
    viol_fields = [
        "activity_nr", "citation_id", "delete_flag", "standard",
        "viol_type", "issuance_date", "abate_date", "abate_complete",
        "current_penalty", "initial_penalty", "nr_instances",
        "nr_exposed", "gravity", "emphasis", "hazcat",
        "hazsub1", "hazsub2", "hazsub3", "hazsub4", "hazsub5",
    ]
    viol_count = build_dataset(
        "OSHA_violation", "violations_bulk.csv",
        date_field="issuance_date", cutoff_date=cutoff, fields=viol_fields,
    )
    print(f"  -> {viol_count:,} violations")

    # --- Accidents (all time) ---
    print("\nBuilding accidents (all)...")
    acc_fields = [
        "summary_nr", "report_id", "event_date", "event_time",
        "event_desc", "event_keyword", "fatality", "abstract_text",
    ]
    acc_count = build_dataset(
        "OSHA_accident", "accidents_bulk.csv", fields=acc_fields,
    )
    print(f"  -> {acc_count:,} accidents")

    # --- Accident injuries (all time) ---
    print("\nBuilding accident injuries (all)...")
    inj_fields = [
        "summary_nr", "rel_insp_nr", "age", "sex",
        "nature_of_inj", "part_of_body", "src_of_injury",
        "event_type", "degree_of_inj", "hazsub", "fall_distance",
        "fall_ht",
    ]
    inj_count = build_dataset(
        "OSHA_accident_injury", "accident_injuries_bulk.csv", fields=inj_fields,
    )
    print(f"  -> {inj_count:,} accident injuries")

    # --- Accident abstracts (pre-joined) ---
    abs_count = build_abstracts()

    # --- General Duty narratives (pre-joined) ---
    gd_count = build_gen_duty_narratives()

    # --- Meta ---
    with open(META_FILE, "w") as f:
        json.dump({
            "date": str(date.today()),
            "source": "local_download",
            "inspections": insp_count,
            "violations": viol_count,
            "accidents": acc_count,
            "accident_injuries": inj_count,
            "accident_abstracts": abs_count,
            "gen_duty_narratives": gd_count,
            "complete": insp_count > 0 and viol_count > 0,
        }, f, indent=2)

    # Build SQLite cache from the CSVs we just wrote
    build_sqlite_db()

    # Build company-key index for search
    print("  Building company-key index for search...")
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.search.grouped_search import build_company_key_index, save_company_key_index
    from src.data_retrieval.osha_client import OSHAClient
    _client = OSHAClient()
    _client.ensure_cache()
    _idx = build_company_key_index(_client)
    save_company_key_index(_idx)
    print(f"  Company-key index: {len(_idx[1]):,} unique company keys")

    # Clean stale model so it retrains on fresh data
    for stale in ["risk_model.pkl", "population_data.json"]:
        p = os.path.join(CACHE_DIR, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed stale {stale}")

    # ── Build ML risk model (trains GBR + temporal calibration) ──────────
    print("\nBuilding ML risk model …")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.scoring.ml_risk_scorer import MLRiskScorer
    from datetime import date as _date

    scorer = MLRiskScorer()  # triggers _load_or_build() → trains fresh model

    # ── Build penalty percentile thresholds (training data only) ─────────
    print("\nBuilding NAICS penalty percentiles …")
    _build_penalty_percentiles(scorer)

    # ── Build multi-target model ──────────────────────────────────────────
    print("\nBuilding multi-target probabilistic risk model …")
    _build_multi_target_model(scorer)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  {insp_count:,} inspections, {viol_count:,} violations")
    print(f"  {acc_count:,} accidents, {inj_count:,} injuries")
    print(f"  {abs_count:,} pre-joined abstracts")
    print(f"  {gd_count:,} General Duty narratives")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
