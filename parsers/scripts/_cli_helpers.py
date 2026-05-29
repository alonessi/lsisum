"""Helpers shared by CLI entrypoints (date parsing, sys.path setup)."""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_date(s: str | None) -> _dt.date | None:
    if s is None:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")
