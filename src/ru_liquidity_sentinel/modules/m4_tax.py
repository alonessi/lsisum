"""Module M4 — Tax-period seasonality (TZ output: mad_score + flags + MIO).

The user provided a refreshed dataset (``m4_tax_calendar.csv``,
``m4_tax_flags_daily.csv``) that now spans 2014–2026 and ships full
``events_count`` / ``titles`` for the entire calendar (the previous
file only had populated events from 2021 on).

Per the TZ, M4 is the *seasonality* module.  We expose:

* a MAD-based robust z-score for the **events count** — captures
  "unusually busy tax days" without baking the multiplier into the
  module output;
* the original boolean tax-peak / EoM / EoQ / EoY flags from the
  parsed daily file;
* an aggregate intensity flag ``m4_flag_tax_week`` (peak ±2 days);
* the **MIO** :class:`RobustOnlineCUSUM` accumulator on the events
  count;
* the multiplicative ``m4_seasonal_factor`` ∈ ``[1.0, 1.4]`` (kept as
  a derived feature, no longer applied implicitly inside aggregator).

NOTE: there is **no smoothing**.  The ``flag_has_event`` and ``dom``
columns are passed through as-is and the events-count distribution
is winsorised with the rolling-quantile helper instead of an EWMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import (
    align_to_calendar,
    rolling_mad_zscore,
    robust_online_cusum_series,
)


PEAK_COLS = [
    "flag_tax_peak_15",
    "flag_tax_peak_20",
    "flag_tax_peak_25",
    "flag_tax_peak_28",
]

CAL_COLS = [
    "flag_end_of_month",
    "flag_quarter_end",
    "flag_year_end",
]


def build_m4(
    tax_flags_daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    mad_window_days: int = 365 * 3,
    factor_max: float = 1.4,
) -> pd.DataFrame:
    """Construct the M4 daily feature frame (TZ output: mad + flags + MIO)."""

    df = tax_flags_daily.copy().set_index("date").sort_index()

    out = pd.DataFrame(index=calendar)
    out.index.name = "date"

    rename_map = {
        "flag_tax_peak_15": "m4_flag_tax_peak_15",
        "flag_tax_peak_20": "m4_flag_tax_peak_20",
        "flag_tax_peak_25": "m4_flag_tax_peak_25",
        "flag_tax_peak_28": "m4_flag_tax_peak_28",
        "flag_end_of_month": "m4_flag_end_of_month",
        "flag_quarter_end": "m4_flag_quarter_end",
        "flag_year_end": "m4_flag_year_end",
        "flag_has_event": "m4_flag_has_event",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            out[dst] = align_to_calendar(
                df[src], calendar, method=None, fill_value=0
            ).astype("int8")
        else:
            out[dst] = pd.Series(0, index=calendar, dtype="int8")

    base_peak = (
        out[
            [
                "m4_flag_tax_peak_15",
                "m4_flag_tax_peak_20",
                "m4_flag_tax_peak_25",
                "m4_flag_tax_peak_28",
            ]
        ].sum(axis=1)
        > 0
    ).astype("int8")
    rolling_peak = base_peak.rolling(window=5, center=True, min_periods=1).max()
    out["m4_flag_tax_week"] = rolling_peak.fillna(0).astype("int8")

    events_count = align_to_calendar(
        df.get("events_count", pd.Series(dtype=float)),
        calendar,
        method=None,
        fill_value=0,
    ).astype(float)
    out["m4_events_count"] = events_count

    out["m4_mad_events_count"] = rolling_mad_zscore(
        events_count, window_days=mad_window_days, direction="upper"
    )

    intensity = (
        out[
            [
                "m4_flag_tax_peak_15",
                "m4_flag_tax_peak_20",
                "m4_flag_tax_peak_25",
                "m4_flag_tax_peak_28",
            ]
        ].sum(axis=1)
        + 0.5
        * out[
            [
                "m4_flag_end_of_month",
                "m4_flag_quarter_end",
                "m4_flag_year_end",
            ]
        ].sum(axis=1)
        + out["m4_flag_tax_week"].astype(int)
    ).clip(lower=0)
    if intensity.max() > 0:
        normalised = intensity / float(intensity.max())
    else:
        normalised = intensity.astype(float)
    out["m4_tax_pressure"] = normalised.astype(float)
    out["m4_seasonal_factor"] = (
        1.0 + (factor_max - 1.0) * out["m4_tax_pressure"]
    ).clip(lower=1.0, upper=factor_max)

    out["m4_mio_cusum"] = robust_online_cusum_series(
        events_count.rename("m4"),
        window_size=min(756, mad_window_days),
    ).values
    return out
