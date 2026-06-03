"""Robust normalisation helpers for daily liquidity signals."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


MAD_TO_STD = 1.4826


def rolling_mad_zscore(
    series: pd.Series,
    window_days: int,
    min_periods: int = 30,
    direction: str = "two-sided",
) -> pd.Series:
    """Calculate a rolling robust z-score using median absolute deviation."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex")

    series = series.astype(float).sort_index()
    win = f"{window_days}D"

    median = series.rolling(win, min_periods=min_periods).median()
    abs_dev = (series - median).abs()
    mad = abs_dev.rolling(win, min_periods=min_periods).median()

    denom = MAD_TO_STD * mad
    denom = denom.where(denom > 1e-9, np.nan)
    zscore = (series - median) / denom

    if direction == "upper":
        zscore = zscore.clip(lower=0.0)
    elif direction == "lower":
        zscore = (-zscore).clip(lower=0.0)
    elif direction != "two-sided":
        raise ValueError(f"Unknown direction: {direction}")

    zscore = zscore.clip(lower=-15.0, upper=15.0)

    return zscore.fillna(0.0)


def zscore_to_subindex(zscore: pd.Series, scale: float = 1.0) -> pd.Series:
    """Map a robust z-score to a bounded ``[0, 100]`` sub-index."""

    zscore = zscore.astype(float)
    sub = 100.0 / (1.0 + np.exp(-zscore / scale))
    if (zscore.fillna(0) >= 0).all():
        sub = (sub - 50.0) * 2.0
        sub = sub.clip(lower=0.0, upper=100.0)
    return sub


def smooth_ewma(
    series: pd.Series,
    halflife_days: float = 7.0,
    min_periods: int = 1,
) -> pd.Series:
    """Apply causal time-aware EWMA smoothing."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex")

    s = series.astype(float).sort_index()
    return s.ewm(
        halflife=pd.Timedelta(days=halflife_days),
        times=s.index,
        adjust=True,
        min_periods=min_periods,
    ).mean()


def rolling_percentile_rank(
    series: pd.Series,
    window_days: int,
    min_periods: int = 60,
) -> pd.Series:
    """Calculate rolling empirical-CDF percentile rank in ``[0, 100]``."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex")

    s = series.astype(float).sort_index()
    win = f"{window_days}D"

    def _rank(window: np.ndarray) -> float:
        if window.size == 0:
            return np.nan
        last = window[-1]
        if np.isnan(last):
            return np.nan
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            return np.nan
        return float((valid < last).sum()) / float(valid.size) * 100.0

    rank = s.rolling(win, min_periods=min_periods).apply(_rank, raw=True)
    return rank.fillna(0.0)


def winsorize_rolling(
    series: pd.Series,
    window_days: int,
    upper_q: float = 0.995,
    min_periods: int = 30,
) -> pd.Series:
    """Cap observations at a causal rolling upper quantile."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex")

    s = series.astype(float).sort_index()
    win = f"{window_days}D"
    past = s.shift(1, freq="D")
    cap = past.rolling(win, min_periods=min_periods).quantile(upper_q)
    cap = cap.reindex(s.index, method="ffill")
    return s.where(s.le(cap) | cap.isna(), cap)


def align_to_calendar(
    series: pd.Series,
    calendar: pd.DatetimeIndex,
    method: str = "ffill",
    fill_value: float | None = None,
) -> pd.Series:
    """Reindex a sparse series onto the daily modelling calendar."""

    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.reindex(calendar)
    if method:
        s = s.ffill() if method == "ffill" else s.bfill()
    if fill_value is not None:
        s = s.fillna(fill_value)
    return s


class RobustOnlineCUSUM:
    """Causal robust CUSUM detector with bounded output."""

    def __init__(
        self,
        window_size: int = 756,
        threshold: float = 10.0,
        drift: float = 1.0,
        cooldown_factor: float = 0.5,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.drift = drift
        self.cooldown_factor = cooldown_factor
        self.cusum = 0.0
        self.history: "deque[float]" = deque(maxlen=window_size)

    def update(self, current_value: float) -> float:
        if len(self.history) < 21:
            self.history.append(current_value)
            return 0.0

        hist_array = np.array(self.history, dtype=float)
        rolling_median = float(np.median(hist_array))
        rolling_mad = float(np.median(np.abs(hist_array - rolling_median)))
        if rolling_mad == 0:
            rolling_mad = 1e-6

        z_robust = (current_value - rolling_median) / (rolling_mad * MAD_TO_STD)
        self.history.append(current_value)

        if z_robust < 0:
            self.cusum *= self.cooldown_factor

        self.cusum = max(0.0, self.cusum + z_robust - self.drift)

        return float(min(1.0, self.cusum / self.threshold))


def robust_online_cusum_series(
    series: pd.Series,
    window_size: int = 756,
    threshold: float = 10.0,
    drift: float = 1.0,
    cooldown_factor: float = 0.5,
) -> pd.Series:
    """Run :class:`RobustOnlineCUSUM` over a pandas Series."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex")

    detector = RobustOnlineCUSUM(
        window_size=window_size,
        threshold=threshold,
        drift=drift,
        cooldown_factor=cooldown_factor,
    )
    out = np.zeros(len(series), dtype=float)
    values = series.astype(float).values
    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = 0.0
            continue
        out[i] = detector.update(float(v))
    return pd.Series(out, index=series.index, name=f"{series.name or 'mio'}_cusum")
