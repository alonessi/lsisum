"""Common helpers shared across module parsers."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

_NBSP = "\u00a0"
_NUMERIC_PATTERN = re.compile(r"[\s\u00a0]+")


def parse_ru_number(value: object) -> float | None:
    """Parse Russian-formatted numbers ('1 234,56' -> 1234.56). Returns None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s in {"-", "—", "–", "–", "n/a", "N/A", "x", "**", "***"}:
        return None
    s = s.replace(_NBSP, "").replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-+eE]", "", s)
    if not s or s in {".", "-", "+"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ru_date(value: object) -> pd.Timestamp | None:
    """Parse 'DD.MM.YYYY' or ISO date into pandas Timestamp; returns NaT on failure."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s, dayfirst=True))
    except (ValueError, TypeError):
        return None


def coerce_numeric_columns(
    df: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = df[c].map(parse_ru_number)
    return df


def write_csv(df: pd.DataFrame, path) -> None:
    """Write a CSV with UTF-8 BOM for friendliness with Excel + ISO dates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
