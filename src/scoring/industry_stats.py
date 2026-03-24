import pandas as pd
from typing import Dict, Optional

from src.data_retrieval.naics_lookup import get_industry_name, load_naics_map


def compute_industry_stats(
    df: pd.DataFrame,
    min_sample: int = 10,
    naics_map: Optional[dict] = None,
) -> Dict[str, dict]:
    if naics_map is None:
        naics_map = load_naics_map()

    df = df.copy()
    df["industry_group"] = df["industry_group"].fillna("").astype(str)

    stats: Dict[str, dict] = {}

    for digits in (4, 3, 2):
        df["_grp"] = df["industry_group"].apply(
            lambda g: g[:digits] if len(g) >= digits else ""
        )
        for grp, grp_df in df[df["_grp"] != ""].groupby("_grp"):
            if grp in stats or len(grp_df) < min_sample:
                continue
            vpi_series = grp_df["raw_vpi"].astype(float)
            pen_series = grp_df["raw_avg_pen"].astype(float)
            sr_series = grp_df["raw_serious_rate"].astype(float)
            wr_series = grp_df["raw_wr_rate"].astype(float)
            stats[grp] = {
                "label":              get_industry_name(grp, naics_map),
                "count":              int(len(grp_df)),
                "avg_violation_rate": float(vpi_series.mean()),
                "std_violation_rate": float(vpi_series.std(ddof=0)) or 1e-6,
                "avg_penalty":        float(pen_series.mean()),
                "std_penalty":        float(pen_series.std(ddof=0)) or 1e-6,
                "avg_serious_ratio":  float(sr_series.mean()),
                "std_serious_ratio":  float(sr_series.std(ddof=0)) or 1e-6,
                "avg_willful_repeat": float(wr_series.mean()),
                "std_willful_repeat": float(wr_series.std(ddof=0)) or 1e-6,
            }

    return stats


def compute_relative_features(
    company_row: dict,
    industry_stats: Dict[str, dict],
    naics_map: Optional[dict] = None,
    min_sample: int = 10,
) -> dict:
    if naics_map is None:
        naics_map = load_naics_map()

    raw_group = str(company_row.get("industry_group") or "").strip()
    missing_naics = not raw_group

    industry_entry = None
    resolved_group = raw_group or None
    for grp_len in (4, 3, 2):
        if not raw_group or len(raw_group) < grp_len:
            continue
        key = raw_group[:grp_len]
        entry = industry_stats.get(key)
        if entry and entry.get("count", 0) >= min_sample:
            industry_entry = entry
            resolved_group = key
            break

    if industry_entry is None:
        _nan = float("nan")
        return {
            "relative_violation_rate": _nan,
            "relative_penalty":        _nan,
            "relative_serious_ratio":  _nan,
            "relative_willful_repeat": _nan,
            "industry_group":          resolved_group,
            "industry_label":          get_industry_name(raw_group, naics_map),
            "industry_count":          0,
            "missing_naics":           missing_naics,
        }

    def _z(val: float, avg: float, std: float) -> float:
        return (float(val) - avg) / max(abs(std), 1e-6)

    return {
        "relative_violation_rate": _z(
            company_row.get("raw_vpi", 0.0),
            industry_entry["avg_violation_rate"],
            industry_entry["std_violation_rate"],
        ),
        "relative_penalty": _z(
            company_row.get("raw_avg_pen", 0.0),
            industry_entry["avg_penalty"],
            industry_entry["std_penalty"],
        ),
        "relative_serious_ratio": _z(
            company_row.get("raw_serious_rate", 0.0),
            industry_entry["avg_serious_ratio"],
            industry_entry["std_serious_ratio"],
        ),
        "relative_willful_repeat": _z(
            company_row.get("raw_wr_rate", 0.0),
            industry_entry["avg_willful_repeat"],
            industry_entry["std_willful_repeat"],
        ),
        "industry_group":  resolved_group,
        "industry_label":  industry_entry.get("label", "Unknown Industry"),
        "industry_count":  industry_entry.get("count", 0),
        "missing_naics":   missing_naics,
    }
