"""Module M1 — Reserve averaging + RUONIA spread.

Per the TZ, this module produces (no smoothing, no sub-index):

* MAD-based robust z-scores of every primary signal
  (3-year rolling window, ``(x - median) / (1.4826 * MAD)``).
* Binary flags marking known calendar effects (end of averaging period).
* The **MIO** feature — :class:`RobustOnlineCUSUM` accumulator on the
  primary signal (reserve spread).  Strictly causal, in ``[0, 1]``.

We expose the resulting daily DataFrame indexed by ``date`` with the
columns:

* ``m1_spread_blnrub``               — daily-aligned monthly spread.
* ``m1_actual_balances_blnrub``      — actual balances on accounts.
* ``m1_required_avg_blnrub``         — required averaging amount.
* ``m1_ruonia_spread_pct``           — RUONIA − key rate.
* ``m1_mad_spread``                  — robust z-score (upper-tail) of the spread.
* ``m1_mad_ruonia_spread``           — robust z-score (upper-tail) of (RUONIA − key rate).
* ``m1_flag_end_of_period``          — 1 in the last ``end_of_period_days`` calendar days of the averaging period.
* ``m1_mio_cusum``                   — :class:`RobustOnlineCUSUM` accumulator on ``m1_spread_blnrub``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import (
    align_to_calendar,
    rolling_mad_zscore,
    robust_online_cusum_series,
    winsorize_rolling,
)


def _expand_periods_to_daily(
    rreserves: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Forward-fill monthly reserve series to a daily calendar."""

    df = rreserves.copy().set_index("date").sort_index()

    cols = [
        "actual_balances_blnrub",
        "required_avg_blnrub",
        "spread_blnrub",
    ]
    daily = pd.DataFrame(index=calendar)
    for col in cols:
        daily[col] = align_to_calendar(df[col], calendar, method="ffill")
    return daily


def _end_of_period_flag(
    rreserves: pd.DataFrame, calendar: pd.DatetimeIndex, days: int = 5
) -> pd.Series:
    """Mark the last ``days`` calendar days of each averaging period."""

    starts = rreserves["date"].sort_values().reset_index(drop=True)
    flags = pd.Series(0, index=calendar, dtype="int8")
    if starts.empty:
        return flags

    starts_list = list(starts) + [calendar.max() + pd.Timedelta(days=1)]
    for cur, nxt in zip(starts_list[:-1], starts_list[1:]):
        end = nxt - pd.Timedelta(days=1)
        window_start = end - pd.Timedelta(days=days - 1)
        mask = (calendar >= window_start) & (calendar <= end)
        flags.loc[mask] = 1
    return flags


def build_m1(
    rreserves: pd.DataFrame,
    ruonia: pd.DataFrame,
    keyrate: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    mad_window_days: int,
    end_of_period_days: int = 5,
) -> pd.DataFrame:
    """Construct the M1 daily feature frame (mad scores + flags + MIO)."""

    daily = _expand_periods_to_daily(rreserves, calendar)

    ruonia_aligned = align_to_calendar(
        ruonia.set_index("date")["ruonia_rate_pct"], calendar, method="ffill"
    )
    keyrate_aligned = align_to_calendar(
        keyrate.set_index("date")["key_rate_pct"], calendar, method="ffill"
    )

    out = pd.DataFrame(index=calendar)
    out.index.name = "date"
    out["m1_spread_blnrub"] = daily["spread_blnrub"]
    out["m1_actual_balances_blnrub"] = daily["actual_balances_blnrub"]
    out["m1_required_avg_blnrub"] = daily["required_avg_blnrub"]
    out["m1_ruonia_spread_pct"] = ruonia_aligned - keyrate_aligned

    # EDA-driven winsorisation of the reserve spread (skew=3.1, ex.kurt=15.7,
    # p99/p50≈3.6).  Causal rolling 1-year p99.5 cap so a one-off transition
    # month does not bias the 3-year MAD denominator.
    out["m1_spread_winsorised"] = winsorize_rolling(
        out["m1_spread_blnrub"], window_days=365, upper_q=0.995
    )
    out["m1_mad_spread"] = rolling_mad_zscore(
        out["m1_spread_winsorised"], window_days=mad_window_days, direction="upper"
    )
    out["m1_mad_ruonia_spread"] = rolling_mad_zscore(
        out["m1_ruonia_spread_pct"], window_days=mad_window_days, direction="upper"
    )
    out["m1_flag_end_of_period"] = _end_of_period_flag(
        rreserves, calendar, days=end_of_period_days
    ).astype("int8")

    # MIO — RobustOnlineCUSUM on the (winsorised) reserve spread.
    out["m1_mio_cusum"] = robust_online_cusum_series(
        out["m1_spread_winsorised"].rename("m1"),
        window_size=min(756, mad_window_days),
    ).values
    return out
