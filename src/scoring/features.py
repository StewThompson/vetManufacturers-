"""features.py — Raw feature extraction for OSHA establishment records.

Extracts the 25-element absolute feature vector from a list of OSHARecord
objects.  This is the core per-establishment aggregation step shared by
both the online scoring path and the offline population-building path.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta
from typing import List, Optional, Tuple

import numpy as np

from src.models.osha_record import OSHARecord


def extract_establishment_features_raw(
    records: List[OSHARecord],
) -> Tuple[list, Optional[str], float, float, float, float]:
    """Extract 25 absolute features for one establishment.

    Returns ``(features_25, naics_code, vpi, avg_pen, raw_serious_rate, raw_wr_rate)``.
    Features use simple per-inspection rates so train and inference are aligned.
    Recency is captured by the ``recent_ratio`` feature (3-year window).
    """
    one_year_ago = date.today() - timedelta(days=1095)  # 3-year recency window
    n_insp = len(records)
    recent = 0
    severe = 0
    clean = 0
    n_viols = 0
    serious_raw = 0
    willful_raw = 0
    repeat_raw = 0
    total_pen = 0.0
    all_penalties: list = []
    acc_count = 0
    fat_count = 0
    inj_count = 0
    gravities: list = []
    time_adj_pen = 0.0
    max_insp_pen = 0.0
    estab_sizes: list = []
    recent_viol_count = 0
    recent_wr_raw = 0

    naics_votes: Counter = Counter()

    for r in records:
        viols = r.violations
        n_viols += len(viols)

        serious_raw += sum(1 for v in viols if v.severity == "Serious")
        willful_raw += sum(1 for v in viols if v.is_willful)
        repeat_raw += sum(1 for v in viols if v.is_repeat)

        pens = [v.penalty_amount for v in viols]
        insp_pen = sum(pens)
        if insp_pen > max_insp_pen:
            max_insp_pen = insp_pen
        total_pen += insp_pen
        all_penalties.extend(pens)
        age_years = max(0.0, (date.today() - r.date_opened).days / 365.25)
        time_adj_pen += insp_pen * math.exp(-age_years / 3.0)

        if r.date_opened >= one_year_ago:
            recent += 1
            recent_viol_count += len(viols)
            recent_wr_raw += sum(1 for v in viols if v.is_willful or v.is_repeat)

        if r.accidents:
            severe += 1
            for a in r.accidents:
                acc_count += 1
                if a.fatality:
                    fat_count += 1
                inj_count += len(a.injuries)

        for v in viols:
            if v.gravity:
                try:
                    gravities.append(float(v.gravity))
                except (ValueError, TypeError):
                    pass

        if not viols:
            clean += 1

        if r.naics_code:
            naics_votes[r.naics_code] += 1

        if r.nr_in_estab:
            try:
                sz = float(r.nr_in_estab)
                if sz > 0:
                    estab_sizes.append(sz)
            except (ValueError, TypeError):
                pass

    naics_code = naics_votes.most_common(1)[0][0] if naics_votes else None

    avg_pen = float(np.mean(all_penalties)) if all_penalties else 0.0
    max_pen = max(all_penalties) if all_penalties else 0.0
    recent_ratio = recent / n_insp if n_insp else 0.0
    vpi = n_viols / n_insp if n_insp else 0.0
    avg_gravity = float(np.mean(gravities)) if gravities else 0.0
    pen_per_insp = total_pen / n_insp if n_insp else 0.0
    clean_ratio = clean / n_insp if n_insp else 0.0
    serious_rate = serious_raw / n_insp if n_insp else 0.0
    willful_rate = willful_raw / n_insp if n_insp else 0.0
    repeat_rate = repeat_raw / n_insp if n_insp else 0.0
    severe_rate = severe / n_insp if n_insp else 0.0
    acc_rate = acc_count / n_insp if n_insp else 0.0
    fat_rate = fat_count / n_insp if n_insp else 0.0
    inj_rate = inj_count / n_insp if n_insp else 0.0

    raw_serious_rate = serious_raw / max(n_viols, 1)
    raw_wr_rate = (willful_raw + repeat_raw) / max(n_viols, 1)

    total_wr = willful_raw + repeat_raw
    recent_wr_rate = recent_wr_raw / max(total_wr, 1)
    vpi_recent = recent_viol_count / max(recent, 1)
    trend_delta = vpi - vpi_recent

    median_estab_size = float(np.median(estab_sizes)) if estab_sizes else 0.0

    features_25 = [
        n_insp, n_viols, serious_rate, willful_rate, repeat_rate,
        total_pen, avg_pen, max_pen, recent_ratio, severe_rate, vpi,
        acc_rate, fat_rate, inj_rate, avg_gravity,
        pen_per_insp, clean_ratio,
        time_adj_pen,
        recent_wr_rate,
        trend_delta,
        math.log1p(willful_raw),
        math.log1p(repeat_raw),
        1.0 if fat_count > 0 else 0.0,
        math.log1p(max_insp_pen),
        math.log1p(median_estab_size),
    ]

    return features_25, naics_code, vpi, avg_pen, raw_serious_rate, raw_wr_rate
