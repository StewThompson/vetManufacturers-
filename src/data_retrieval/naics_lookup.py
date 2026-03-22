"""
NAICS code â†’ industry name lookup.

Reads exclusively from the 2022 NAICS Excel file located at:
  src/data_retrieval/naics_data/2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx

Public API
----------
    load_naics_map(xlsx_path=None) -> dict[str, str]
    get_industry_name(naics_code, naics_map) -> str
"""

from __future__ import annotations

import os
import re
from typing import Optional


# ------------------------------------------------------------------ #
#  Module-level lazy singleton
# ------------------------------------------------------------------ #

_NAICS_MAP: Optional[dict] = None

_DATA_DIR = os.path.join(os.path.dirname(__file__), "naics_data")
_XLSX_NAME = "2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx"


def load_naics_map(xlsx_path: Optional[str] = None) -> dict[str, str]:
    """Return the NAICS code â†’ title mapping (lazy singleton).

    Loads from the standard 2022 NAICS Excel file in the naics_data/ directory.
    Raises RuntimeError if the file is missing.
    """
    global _NAICS_MAP
    if _NAICS_MAP is not None:
        return _NAICS_MAP

    target = xlsx_path or os.path.join(_DATA_DIR, _XLSX_NAME)
    if not os.path.exists(target):
        raise RuntimeError(
            f"NAICS Excel file not found: {target}\n"
            "Place '2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx' "
            "in src/data_retrieval/naics_data/"
        )

    naics_map: dict[str, str] = {}
    try:
        import pandas as pd
        df = pd.read_excel(target, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        code_col = next(
            (c for c in df.columns if "code" in c and "naics" in c), None
        )
        title_col = next(
            (c for c in df.columns if "title" in c), None
        )
        if code_col and title_col:
            for _, row in df.dropna(subset=[code_col, title_col]).iterrows():
                code = re.sub(r"[^0-9]", "", str(row[code_col]).strip())
                title = str(row[title_col]).strip()
                if 2 <= len(code) <= 6 and title:
                    naics_map[code] = title
            print(f"NAICS lookup: loaded {len(naics_map)} codes from Excel.")
        else:
            raise ValueError(f"Could not find NAICS code/title columns in {target}")
    except Exception as exc:
        raise RuntimeError(f"NAICS Excel load failed: {exc}") from exc

    _NAICS_MAP = naics_map
    return _NAICS_MAP


def get_industry_name(naics_code: Optional[str], naics_map: Optional[dict] = None) -> str:
    """Return the human-readable industry name for a NAICS code.

    Tries 6â†’5â†’4â†’3â†’2-digit prefix matches so that a 6-digit code such as
    ``332710`` resolves to the nearest available title.  Returns
    ``"Unknown Industry"`` if no match is found.
    """
    if not naics_code:
        return "Unknown Industry"
    if naics_map is None:
        naics_map = load_naics_map()

    digits = re.sub(r"[^0-9]", "", str(naics_code).strip())
    if not digits:
        return "Unknown Industry"

    for length in (6, 5, 4, 3, 2):
        prefix = digits[:length]
        if prefix in naics_map:
            return naics_map[prefix]

    return "Unknown Industry"
