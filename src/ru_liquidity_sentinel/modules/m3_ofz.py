"""Module M3 — OFZ primary auctions (Minfin)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import (
    align_to_calendar,
    rolling_mad_zscore,
    robust_online_cusum_series
)

def build_m3(
    ofz_auctions: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    mad_window_days: int,
) -> pd.DataFrame:
    """Construct the M3 daily feature frame (no smoothing)."""

    df = ofz_auctions.copy()

    daily = (
        df.groupby("date", as_index=True)
        .agg(
            offer_volume_mlnrub=("offer_volume_mlnrub", "sum"),
            demand_volume_mlnrub=("demand_volume_mlnrub", "sum"),
            allotment_volume_mlnrub=("allotment_volume_mlnrub", "sum"),
            weighted_avg_yield_pct=("weighted_avg_yield_pct", "mean"),
            cover_ratio=("cover_ratio", "mean"),
        )
        .sort_index()
    )

    safe_offer = daily["offer_volume_mlnrub"].replace({0.0: np.nan})
    cover_recomputed = daily["demand_volume_mlnrub"] / safe_offer
    daily["cover_ratio_clean"] = cover_recomputed.fillna(daily["cover_ratio"])

    out = pd.DataFrame(index=calendar)
    out.index.name = "date"
    
    out["m3_cover_ratio"] = align_to_calendar(
        daily["cover_ratio_clean"], calendar, method="ffill"
    )
    out["m3_yield_pct"] = align_to_calendar(
        daily["weighted_avg_yield_pct"], calendar, method="ffill"
    )

    rolling_med = out["m3_yield_pct"].rolling("60D", min_periods=10).median()
    out["m3_yield_spread_pct"] = (out["m3_yield_pct"] - rolling_med)

    out["m3_mad_cover_low"] = rolling_mad_zscore(
        out["m3_cover_ratio"], window_days=mad_window_days, direction="lower"
    )
    out["m3_mad_yield_spread"] = rolling_mad_zscore(
        out["m3_yield_spread_pct"], window_days=mad_window_days, direction="upper"
    )

    out["m3_flag_undersubscribed"] = (out["m3_cover_ratio"] < 1.2).astype("int8")
    out["m3_flag_oversubscribed"] = pd.Series(0, index=calendar, dtype="int8")

    cover_for_cusum = (-out["m3_cover_ratio"].fillna(0.0)).rename("m3")
    out["m3_mio_cusum"] = robust_online_cusum_series(
        cover_for_cusum, window_size=min(756, mad_window_days)
    ).values
    return out
