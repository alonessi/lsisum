"""M4 feature builder: tax-calendar seasonality."""

from __future__ import annotations

import pandas as pd

from ..normalize import (
    align_to_calendar,
    rolling_mad_zscore,
    robust_online_cusum_series,
)


def build_m4(
    tax_flags_daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    mad_window_days: int = 365 * 3,
    factor_max: float = 1.4,
) -> pd.DataFrame:
    """Build daily M4 seasonality features from tax-calendar flags."""

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

    peak_cols = [
        "m4_flag_tax_peak_15",
        "m4_flag_tax_peak_20",
        "m4_flag_tax_peak_25",
        "m4_flag_tax_peak_28",
    ]
    calendar_cols = [
        "m4_flag_end_of_month",
        "m4_flag_quarter_end",
        "m4_flag_year_end",
    ]

    base_peak = (out[peak_cols].sum(axis=1) > 0).astype("int8")
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
        out[peak_cols].sum(axis=1)
        + 0.5 * out[calendar_cols].sum(axis=1)
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
