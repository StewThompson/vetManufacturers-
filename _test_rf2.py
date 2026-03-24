import sqlite3, os, sys, re, time

sys.path.insert(0, '.')
from src.search.grouped_search import group_establishments, normalize_establishment_name

db = os.path.join('ml_cache', 'osha_cache.db')
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT DISTINCT estab_name FROM inspections "
    "WHERE estab_name IS NOT NULL AND estab_name != '' "
    "ORDER BY estab_name"
).fetchall()
all_names = [r["estab_name"].strip().title() for r in rows if r["estab_name"]]
print(f"Total raw estab names: {len(all_names)}")

class MockOSHA:
    _use_sqlite = True
    def __init__(self, c):
        self._db_conn = c
    def _db_rows(self, sql, params=()):
        cur = self._db_conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    def company_match_key(self, name):
        s = name.strip().upper()
        s = re.sub(r'^(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,}\s+-\s+', '', s)
        s = re.sub(r'\s+D/?B/?A\s+.+$', '', s)
        for _ in range(3):
            new = re.sub(r'[,.]?\s*\b(?:INC\.?|LLC\.?|L\.?P\.?|CORP\.?|CO\.?|LTD\.?|LIMITED|INCORPORATED|CORPORATION|COMPANY|GRP|GROUP)\s*$', '', s, flags=re.IGNORECASE).strip()
            if new == s: break
            s = new
        s = s.rstrip('.,;-\u2013 ').strip()
        s = re.sub(r'\s*&\s*', ' AND ', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        return s

client = MockOSHA(conn)

for query in ["walmart", "walmart supercenter", "fastenal"]:
    t0 = time.time()
    result = group_establishments(query, all_names, client)
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Query: '{query}' ({elapsed:.1f}s)")
    if result.top_group:
        g = result.top_group
        print(f"  TOP: {g.parent_name} | {g.total_facilities} fac | {g.confidence_label} ({g.confidence:.2f})")
        print(f"       high={len(g.high_confidence)} med={len(g.medium_confidence)} low={len(g.low_confidence)}")
        for f in g.all_facilities[:8]:
            print(f"    {f.raw_name} | {f.confidence:.2f} {f.confidence_label}")
        if g.total_facilities > 8:
            print(f"    ... +{g.total_facilities - 8} more")
    for i, og in enumerate(result.other_groups):
        print(f"  OTHER[{i}]: {og.parent_name} | {og.total_facilities} fac | {og.confidence_label}")

conn.close()
