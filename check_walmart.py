import sqlite3
import os

db = os.path.join('ml_cache', 'osha_cache.db')
if not os.path.exists(db):
    print('DB not found at', db)
    print('Files in ml_cache:')
    for f in os.listdir('ml_cache'):
        print(' ', f)
else:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    total = conn.execute('SELECT COUNT(*) FROM inspections').fetchone()[0]
    print(f'Total inspections in DB: {total}')

    # See all distinct company_keys containing WALMART
    walmart_keys = conn.execute(
        "SELECT DISTINCT company_key FROM inspections WHERE company_key LIKE '%WALMART%' ORDER BY company_key"
    ).fetchall()
    print(f'\nDistinct company_keys containing WALMART ({len(walmart_keys)}):')
    for r in walmart_keys:
        print(' ', r[0])

    # Count per key
    print('\nInspection counts per Walmart company_key:')
    counts = conn.execute(
        "SELECT company_key, COUNT(*) as cnt FROM inspections "
        "WHERE company_key LIKE '%WALMART%' GROUP BY company_key ORDER BY cnt DESC"
    ).fetchall()
    grand = 0
    for r in counts:
        print(f'  {r["company_key"]}: {r["cnt"]}')
        grand += r["cnt"]
    print(f'\nTotal Walmart inspections (all keys): {grand}')

    # Also check raw estab_name variants
    print('\nSample raw estab_names for company_key=WALMART (first 20):')
    estabs = conn.execute(
        "SELECT DISTINCT estab_name FROM inspections WHERE company_key='WALMART' LIMIT 20"
    ).fetchall()
    for r in estabs:
        print(' ', r[0])

    # Check what grouped_search would produce — what key does normalize produce for "Walmart"?
    import sys
    sys.path.insert(0, '.')
    from src.data_retrieval.osha_client import OSHAClient
    cl = OSHAClient.__new__(OSHAClient)
    test_inputs = ['Walmart', 'Walmart, Inc', 'WALMART', 'Walmart Warehouse']
    print('\nNormalization results:')
    for t in test_inputs:
        print(f'  "{t}" -> "{cl._normalize_company_name(t.upper())}"')

    conn.close()
