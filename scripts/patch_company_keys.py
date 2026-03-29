"""
patch_company_keys.py
---------------------
Re-computes the company_key column for every row in the inspections table
using the current company_match_key() normalization, then rebuilds indexes
and refreshes the company_key_index.json cache.

Run while the API server is stopped so the DB is not locked.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
from src.data_retrieval.normalization.company_names import company_match_key

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_cache")
DB_PATH   = os.path.join(CACHE_DIR, "osha_cache.db")
IDX_PATH  = os.path.join(CACHE_DIR, "company_key_index.json")

def run():
    print(f"Opening {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")

    # How many rows?
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    print(f"  {total:,} inspection rows to patch")

    # Fetch all distinct (rowid, estab_name) pairs
    print("  Loading distinct estab_name values ...")
    rows = conn.execute(
        "SELECT DISTINCT estab_name FROM inspections WHERE estab_name IS NOT NULL AND estab_name != ''"
    ).fetchall()
    print(f"  {len(rows):,} distinct estab_name values")

    # Build a mapping: estab_name -> new company_key
    mapping = {}
    for (raw,) in rows:
        new_key = company_match_key(raw.upper()).upper()
        mapping[raw] = new_key

    # Spot-check
    sample_raw = list(mapping.keys())[:5]
    print("\n  Sample remappings:")
    for r in sample_raw:
        print(f"    {repr(r[:60]).ljust(65)} -> {repr(mapping[r])}")

    # Apply updates in batches
    print(f"\n  Applying {len(mapping):,} company_key updates ...")
    updated = 0
    batch = []
    for raw, new_key in mapping.items():
        batch.append((new_key, raw))
        if len(batch) >= 5000:
            conn.executemany(
                "UPDATE inspections SET company_key = ? WHERE estab_name = ?",
                batch,
            )
            updated += len(batch)
            conn.commit()
            batch = []
            if updated % 50000 == 0:
                print(f"    {updated:,} estab_names updated ...")
    if batch:
        conn.executemany(
            "UPDATE inspections SET company_key = ? WHERE estab_name = ?",
            batch,
        )
        conn.commit()
        updated += len(batch)

    print(f"  {updated:,} distinct estab_names patched")

    # Rebuild company_names table
    print("  Rebuilding company_names table ...")
    conn.execute("DELETE FROM company_names")
    conn.execute(
        "INSERT OR IGNORE INTO company_names (company_key, name) "
        "SELECT DISTINCT company_key, company_key FROM inspections "
        "WHERE company_key IS NOT NULL AND company_key != ''"
    )
    conn.commit()

    # Verify
    cnt = conn.execute("SELECT COUNT(DISTINCT company_key) FROM inspections").fetchone()[0]
    print(f"  Now {cnt:,} distinct company_keys (was many more before)")

    # Spot-check Walmart
    print("\n  Walmart company_keys after patch:")
    wm_rows = conn.execute(
        "SELECT company_key, COUNT(*) as n FROM inspections "
        "WHERE company_key LIKE '%WALMART%' OR company_key LIKE '%WAL-MART%' "
        "GROUP BY company_key ORDER BY n DESC LIMIT 15"
    ).fetchall()
    for ck, n in wm_rows:
        print(f"    {repr(str(ck)[:70]).ljust(75)}  {n} rows")

    conn.close()

    # Rebuild the company_key_index.json
    print("\n  Rebuilding company_key_index.json ...")
    if os.path.exists(IDX_PATH):
        os.remove(IDX_PATH)
    from src.data_retrieval.osha_client import OSHAClient
    from src.search.grouped_search import build_company_key_index, save_company_key_index
    client = OSHAClient()
    client.ensure_cache()
    idx = build_company_key_index(client)
    save_company_key_index(idx)
    n_keys = len(idx[1])
    # Show Walmart summary in new index
    wm_idx = {k: v for k, v in idx[0].items() if 'WALMART' in k.upper() or 'WAL-MART' in k.upper()}
    print(f"  Index rebuilt: {n_keys:,} unique company keys")
    print(f"\n  Walmart keys in new index:")
    for k in sorted(wm_idx):
        print(f"    {repr(k).ljust(40)} -> {len(wm_idx[k])} estabs")

    print("\nDone.")

if __name__ == "__main__":
    run()
