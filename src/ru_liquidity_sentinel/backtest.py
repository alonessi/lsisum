from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EpisodeReport:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    mean_lsi: float
    max_lsi: float
    days_red: int
    days_yellow: int
    detected: bool


def evaluate_episodes(
    lsi: pd.Series,
    episodes: List[Tuple[str, str, str]],
    red_threshold: float = 70.0,
    yellow_threshold: float = 40.0,
) -> pd.DataFrame:
    """Per-episode summary used for the backtest table on the dashboard."""

    rows: List[Dict] = []
    for name, start, end in episodes:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        window = lsi.loc[(lsi.index >= s) & (lsi.index <= e)]
        if window.empty:
            rows.append(
                {
                    "episode": name,
                    "start": s,
                    "end": e,
                    "days": 0,
                    "mean_lsi": np.nan,
                    "max_lsi": np.nan,
                    "days_red": 0,
                    "days_yellow": 0,
                    "detected": False,
                }
            )
            continue
        days_red = int((window > red_threshold).sum())
        days_yellow = int(((window > yellow_threshold) & (window <= red_threshold)).sum())
        rows.append(
            {
                "episode": name,
                "start": s,
                "end": e,
                "days": int(window.shape[0]),
                "mean_lsi": float(window.mean()),
                "max_lsi": float(window.max()),
                "days_red": days_red,
                "days_yellow": days_yellow,
                "detected": days_red > 0 or days_yellow >= 5,
            }
        )
    return pd.DataFrame(rows)