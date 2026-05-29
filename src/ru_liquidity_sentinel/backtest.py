"""Backtest helpers for historical stress episodes.

The TZ requires:

* a backtest on three known stress episodes — December 2014,
  February–March 2022 and August 2023;
* a hold-out check that the system has *not* learned a historical
  pattern (i.e. evaluate metrics on a chronological out-of-sample
  window);
* sensitivity analysis (handled in ``aggregate.sensitivity_analysis``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score


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


def holdout_metrics(
    lsi: pd.Series,
    proxy_target: pd.Series,
    holdout_start: str = "2024-01-01",
) -> Dict[str, float]:
    """Out-of-sample agreement between the LSI and the proxy target.

    We treat the proxy target as a continuous "stress" reference and
    report MAE between LSI and target on data after ``holdout_start``,
    plus the AUC of using LSI to discriminate "high stress" days
    (target > 70) on the same window.
    """

    df = pd.DataFrame({"lsi": lsi, "target": proxy_target}).dropna()
    df = df.loc[df.index >= pd.Timestamp(holdout_start)]
    if df.empty:
        return {"mae": np.nan, "auc": np.nan, "n": 0}
    mae = float(mean_absolute_error(df["target"], df["lsi"]))
    high_stress = (df["target"] > 70.0).astype(int)
    if high_stress.nunique() > 1:
        auc = float(roc_auc_score(high_stress, df["lsi"]))
    else:
        auc = np.nan
    return {"mae": mae, "auc": auc, "n": int(df.shape[0])}
