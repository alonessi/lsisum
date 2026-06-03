"""Module M2 — CBR repo auctions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import (
    align_to_calendar,
    rolling_mad_zscore,
    robust_online_cusum_series,
)

PRIMARY_TERM_DAYS = 7.0

def build_m2(
    repo_auctions: pd.DataFrame,
    keyrate: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    mad_window_days: int,
) -> pd.DataFrame:
    """Construct the M2 daily feature frame (no smoothing, strictly per TZ)."""

    df = repo_auctions.copy()

    # Рассчитываем взвешенную ставку до группировки по корректной колонке
    df["weighted_rate"] = df["weighted_avg_rate_pct"] * df["allotment_mlnrub"]

    # Aggregate demand (используем bliq_repo_auction_vol), allotment, and rate
    daily_repo = (
        df.groupby(["date", "term_days"], as_index=False)
        .agg(
            demand_volume_mlnrub=("bliq_repo_auction_vol", "sum"),
            allotment_mlnrub=("allotment_mlnrub", "sum"),
            weighted_rate_sum=("weighted_rate", "sum"),
            simple_rate_mean=("weighted_avg_rate_pct", "mean"),
        )
    )

    # Средневзвешенная ставка (откат к простому среднему при нулевом размещении)
    daily_repo["weighted_avg_rate_pct"] = np.where(
        daily_repo["allotment_mlnrub"] > 0,
        daily_repo["weighted_rate_sum"] / daily_repo["allotment_mlnrub"],
        daily_repo["simple_rate_mean"]
    )

    # Calculate Cover Ratio safely: экстремальный стресс (99.0), если ЦБ ничего не дал, но спрос был
    daily_repo["cover_ratio"] = pd.Series(np.where(
        (daily_repo["allotment_mlnrub"] == 0) & (daily_repo["demand_volume_mlnrub"] > 0),
        99.0,
        daily_repo["demand_volume_mlnrub"] / daily_repo["allotment_mlnrub"].replace(0.0, np.nan)
    )).fillna(0.0).values

    primary = (
        daily_repo[daily_repo["term_days"] == PRIMARY_TERM_DAYS]
        .set_index("date")
        .sort_index()
    )

    primary_cover = primary["cover_ratio"].rename("m2_cover_ratio")
    primary_rate = primary["weighted_avg_rate_pct"].rename("m2_rate_pct")

    fine_tuning = (
        daily_repo[daily_repo["term_days"] == 1.0]
        .set_index("date")
        .sort_index()
    )

    keyrate_aligned = align_to_calendar(
        keyrate.set_index("date")["key_rate_pct"], calendar, method="ffill"
    )

    out = pd.DataFrame(index=calendar)
    out.index.name = "date"

    # Fill daily series
    out["m2_cover_ratio"] = align_to_calendar(
        primary_cover, calendar, method="ffill"
    )
    out["m2_rate_pct"] = align_to_calendar(primary_rate, calendar, method="ffill")
    out["m2_rate_spread_pct"] = (out["m2_rate_pct"] - keyrate_aligned).fillna(0.0)

    # MAD scores based on cover ratio and spread
    out["m2_mad_cover"] = rolling_mad_zscore(
        out["m2_cover_ratio"], window_days=mad_window_days, direction="upper"
    )
    out["m2_mad_rate_spread"] = rolling_mad_zscore(
        out["m2_rate_spread_pct"], window_days=mad_window_days, direction="upper"
    )

    # Flag strictly based on cover_ratio > 2.0
    out["m2_flag_high_demand"] = (out["m2_cover_ratio"] > 2.0).astype("int8")

    fine_tuning_flag = pd.Series(0, index=calendar, dtype="int8")
    if not fine_tuning.empty:
        fine_tuning_flag.loc[fine_tuning.index.intersection(calendar)] = 1
        fine_tuning_flag = (
            fine_tuning_flag.rolling("5D").max().fillna(0).astype("int8")
        )
    out["m2_flag_fine_tuning"] = fine_tuning_flag

    # MIO applied to cover ratio
    out["m2_mio_cusum"] = robust_online_cusum_series(
        out["m2_cover_ratio"].rename("m2"),
        window_size=min(756, mad_window_days),
    ).values

    return out