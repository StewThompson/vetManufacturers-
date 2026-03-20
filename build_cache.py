"""
Consolidate downloaded OSHA CSV chunks (OshaData/) into ml_cache/ for fast
in-memory lookup.  Also pre-joins accident abstract lines into single texts.

Usage:
    py build_cache.py              # full build from local OshaData/

Datasets produced in ml_cache/:
    inspections_bulk.csv       – 3-year window, key inspection fields
    violations_bulk.csv        – 3-year window, all violations (incl. zero-penalty)
    accidents_bulk.csv         – all accident events
    accident_injuries_bulk.csv – all injury records (links to inspections)
    accident_abstracts_bulk.csv – pre-joined full-text abstracts (one row per accident)
"""

import os
import csv
import json
import glob
import re
from collections import defaultdict
from datetime import date, timedelta

OSHA_DIR = "OshaData"
CACHE_DIR = "ml_cache"
META_FILE = os.path.join(CACHE_DIR, "bulk_meta.json")
CUTOFF_YEARS = 3


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
    print(f"  Reading {len(files)} chunk(s) from {folder_name}…")
    for i, fp in enumerate(files, 1):
        if i % 20 == 0 or i == len(files):
            print(f"    chunk {i}/{len(files)}…")
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
                print(f"    {count:,} rows…")

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
    print("\nBuilding accident abstracts (pre-joining lines)…")
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

    print(f"  → {len(abstracts):,} unique abstracts")
    return len(abstracts)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cutoff = (date.today() - timedelta(days=CUTOFF_YEARS * 365)).isoformat()

    print("OSHA Cache Builder (local data)")
    print(f"  Source: {OSHA_DIR}/")
    print(f"  Target: {CACHE_DIR}/")
    print(f"  Date cutoff: {cutoff}")

    # --- Inspections (3-year window) ---
    print(f"\nBuilding inspections (open_date >= {cutoff})…")
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
    print(f"  → {insp_count:,} inspections")

    # --- Violations (3-year window, all penalties) ---
    print(f"\nBuilding violations (issuance_date >= {cutoff})…")
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
    print(f"  → {viol_count:,} violations")

    # --- Accidents (all time) ---
    print("\nBuilding accidents (all)…")
    acc_fields = [
        "summary_nr", "report_id", "event_date", "event_time",
        "event_desc", "event_keyword", "fatality", "abstract_text",
    ]
    acc_count = build_dataset(
        "OSHA_accident", "accidents_bulk.csv", fields=acc_fields,
    )
    print(f"  → {acc_count:,} accidents")

    # --- Accident injuries (all time) ---
    print("\nBuilding accident injuries (all)…")
    inj_fields = [
        "summary_nr", "rel_insp_nr", "age", "sex",
        "nature_of_inj", "part_of_body", "src_of_injury",
        "event_type", "degree_of_inj", "hazsub", "fall_distance",
        "fall_ht",
    ]
    inj_count = build_dataset(
        "OSHA_accident_injury", "accident_injuries_bulk.csv", fields=inj_fields,
    )
    print(f"  → {inj_count:,} accident injuries")

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

    # Clean stale model so it retrains on fresh data
    for stale in ["risk_model.pkl", "population_data.json"]:
        p = os.path.join(CACHE_DIR, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed stale {stale}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  {insp_count:,} inspections, {viol_count:,} violations")
    print(f"  {acc_count:,} accidents, {inj_count:,} injuries")
    print(f"  {abs_count:,} pre-joined abstracts")
    print(f"  {gd_count:,} General Duty narratives")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
