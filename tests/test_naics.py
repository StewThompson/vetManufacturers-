import sys, sqlite3, os
sys.path.insert(0, '.')
from src.data_retrieval.osha_client import OSHAClient

db = os.path.join('ml_cache', 'osha_cache.db')
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT company_key, naics_code, COUNT(*) as cnt "
    "FROM inspections WHERE company_key LIKE '%WALMART%' AND naics_code != '' "
    "GROUP BY company_key, naics_code ORDER BY company_key, cnt DESC"
).fetchall()

print(f"{'company_key':<45} {'naics':>10}  count")
print("-" * 65)
for r in rows:
    print(f"{r['company_key']:<45} {r['naics_code']:>10}  {r['cnt']}")

conn.close()
