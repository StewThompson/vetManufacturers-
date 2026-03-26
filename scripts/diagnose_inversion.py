"""Diagnostic script: find why top-5% companies have lower future adverse."""
import sys, csv, numpy as np, math
from collections import defaultdict
from datetime import date, timedelta
sys.path.insert(0, '.')
from src.scoring.pseudo_labeler import pseudo_label

CUTOFF = date(2024, 1, 1)
CACHE_DIR = 'ml_cache'
ONE_YEAR_AGO = date.today() - timedelta(days=365)

print("Loading inspections...")
insp_rows = []
with open(f'{CACHE_DIR}/inspections_bulk.csv', encoding='utf-8', errors='replace') as f:
    for row in csv.DictReader(f):
        insp_rows.append(row)

estab_hist = defaultdict(list)
estab_future = defaultdict(list)
for row in insp_rows:
    name = (row.get('estab_name') or 'UNKNOWN').upper()
    od = row.get('open_date', '')
    try:
        d = date.fromisoformat(od[:10])
    except:
        continue
    if d < CUTOFF:
        estab_hist[name].append(row)
    else:
        estab_future[name].append(row)

print("Loading violations...")
viol_by_act = defaultdict(list)
with open(f'{CACHE_DIR}/violations_bulk.csv', encoding='utf-8', errors='replace') as f:
    for row in csv.DictReader(f):
        if row.get('delete_flag') == 'X':
            continue
        viol_by_act[str(row.get('activity_nr',''))].append(row)

data = []
for name, hist_rows in estab_hist.items():
    if name not in estab_future:
        continue
    n_insp = len(hist_rows)
    viols = []
    fat_count = 0
    for row in hist_rows:
        act = str(row.get('activity_nr',''))
        viols.extend(viol_by_act.get(act, []))
        fat_count += int(row.get('fatalities','0') or 0)
    n_viols = len(viols)
    serious_raw = sum(1 for v in viols if v.get('viol_type') == 'S')
    willful_raw = sum(1 for v in viols if v.get('viol_type') == 'W')
    repeat_raw = sum(1 for v in viols if v.get('viol_type') == 'R')
    pens = [float(v.get('current_penalty') or 0) for v in viols]
    total_pen = sum(pens)
    recent = sum(1 for r in hist_rows
        if len(r.get('open_date',''))>=10
        and date.fromisoformat(r['open_date'][:10]) >= ONE_YEAR_AGO)

    feat = np.zeros(46)
    feat[0] = n_insp
    feat[1] = n_viols
    feat[2] = serious_raw / n_insp
    feat[3] = willful_raw / n_insp
    feat[4] = repeat_raw / n_insp
    feat[5] = total_pen
    feat[6] = float(np.mean(pens)) if pens else 0.0
    feat[7] = max(pens) if pens else 0.0
    feat[8] = recent / n_insp
    feat[9] = 0.0
    feat[10] = n_viols / n_insp
    feat[11] = 0.0
    feat[12] = fat_count / n_insp
    feat[13] = 0.0
    feat[14] = 5.0
    feat[15] = total_pen / n_insp
    feat[16] = sum(1 for r in hist_rows if not viol_by_act.get(str(r.get('activity_nr','')), [])) / n_insp

    score = pseudo_label(feat)

    future_rows = estab_future[name]
    fut_viols = []
    for row in future_rows:
        act = str(row.get('activity_nr',''))
        fut_viols.extend(viol_by_act.get(act, []))
    fut_n = len(future_rows)
    fut_nv = len(fut_viols)
    fut_wr = sum(1 for v in fut_viols if v.get('viol_type') in ('W', 'R'))
    fut_s = sum(1 for v in fut_viols if v.get('viol_type') == 'S')
    fut_pen = sum(float(v.get('current_penalty') or 0) for v in fut_viols)
    fut_fat = sum(int(r.get('fatalities','0') or 0) for r in future_rows)
    adv = (20 * int(fut_fat > 0)
           + (min((fut_fat - 1)*5, 15) if fut_fat > 1 else 0)
           + 8 * int(fut_wr > 0)
           + min(fut_wr * 3, 15)
           + min(fut_s, 10)
           + min(math.log1p(fut_pen) * 0.8, 10)
           + min((fut_nv / fut_n) * 2, 10))

    data.append({
        'name': name, 'score': score, 'adv': adv,
        'n_insp': n_insp, 'fat_rate': fat_count/n_insp,
        'willful_rate': willful_raw/n_insp, 'repeat_rate': repeat_raw/n_insp,
        'vpi': n_viols/n_insp, 'recent': recent/n_insp,
        'serious_rate': serious_raw/n_insp,
    })

scores = np.array([d['score'] for d in data])
advs = np.array([d['adv'] for d in data])
n_insp_arr = np.array([d['n_insp'] for d in data])

p80, p95 = np.percentile(scores, 80), np.percentile(scores, 95)
print(f"p80={p80:.1f}  p95={p95:.1f}")

top5_mask = scores >= p95
mid_mask = (scores >= p80) & (scores < p95)

print(f"\nTop-5%  (n={top5_mask.sum()}, score>={p95:.1f}): mean_adv={advs[top5_mask].mean():.2f}  mean_n_insp={n_insp_arr[top5_mask].mean():.1f}")
print(f"5-20th% (n={mid_mask.sum()}, {p80:.1f}-{p95:.1f}): mean_adv={advs[mid_mask].mean():.2f}  mean_n_insp={n_insp_arr[mid_mask].mean():.1f}")

top5_data = [d for d, m in zip(data, top5_mask) if m]
mid_data  = [d for d, m in zip(data, mid_mask)  if m]

print("\nTop-5% profile (mean per feature):")
for k in ['fat_rate', 'willful_rate', 'repeat_rate', 'serious_rate', 'vpi', 'recent']:
    t5 = np.mean([d[k] for d in top5_data])
    tm = np.mean([d[k] for d in mid_data])
    print(f"  {k:15s}: top5={t5:.3f}  mid={tm:.3f}")

n_insp_top5 = [d['n_insp'] for d in top5_data]
print(f"\nn_insp percentiles for top-5%: P25={np.percentile(n_insp_top5,25):.0f}  P50={np.percentile(n_insp_top5,50):.0f}  P75={np.percentile(n_insp_top5,75):.0f}  P90={np.percentile(n_insp_top5,90):.0f}")
n_insp_mid = [d['n_insp'] for d in mid_data]
print(f"n_insp percentiles for 5-20th%: P25={np.percentile(n_insp_mid,25):.0f}  P50={np.percentile(n_insp_mid,50):.0f}  P75={np.percentile(n_insp_mid,75):.0f}  P90={np.percentile(n_insp_mid,90):.0f}")

print("\nTop-20 highest pseudo-label scores with their future adverse:")
worst = sorted([(d['score'], d['adv'], d['n_insp'], d['fat_rate'], d['willful_rate'], d['vpi'], d['recent'], d['name'][:35]) for d in top5_data], reverse=True)[:20]
print(f"  {'score':>6} {'fut_adv':>8} {'n_insp':>6} {'fat_r':>6} {'wil_r':>6} {'vpi':>5} {'recnt':>5}  name")
for score, adv, ni, fr, wr, vpi, recnt, nm in worst:
    print(f"  {score:6.1f} {adv:8.1f} {ni:6d} {fr:6.3f} {wr:6.3f} {vpi:5.1f} {recnt:5.2f}  {nm}")
