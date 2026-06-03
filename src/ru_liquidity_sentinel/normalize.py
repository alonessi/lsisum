"""MAD-based normalisation helpers.

The TZ requires a robust normalisation of every signal using MAD
(median absolute deviation) over a 3-year rolling window.  The classic
robust z-score formula is used::

    score = (x - median) / (1.4826 * MAD)

For empty / degenerate windows the score is set to ``0`` (i.e. the
signal is treated as neutral when not enough history is available).

Two convenience helpers are also provided here:

* ``mad_zscore_to_subindex`` — squashes a robust z-score into the
  ``[0, 100]`` interval using a logistic transform so it can be used as
  a stress sub-index directly.
* ``align_to_calendar`` — utility that re-indexes a sparse series to a
  daily calendar with forward-fill semantics.

In addition to the rolling-MAD layer, this module exposes the
``RobustOnlineCUSUM`` detector and the convenience wrapper
``robust_online_cusum_series`` that applies it to a daily series.
This is the **MIO** signal — a strictly online, look-ahead-free
stress accumulator that the user requested for every module.
"""

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
    """Robust z-score with rolling median and MAD.

    Parameters
    ----------
    series:
        Input series indexed by ``DatetimeIndex``.
    window_days:
        Rolling window size in calendar days.
    min_periods:
        Minimum number of observations inside the window required to
        produce a non-null score.
    direction:
        ``"two-sided"`` returns the signed z-score, ``"upper"`` clips
        negative values to zero (we only care about the stressed tail
        of a signal — e.g. high cover ratio for repo, large positive
        spread for reserves), ``"lower"`` clips positive values to
        zero (e.g. cover ratio for OFZ where *low* values mean stress).
    """

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
    """Map a robust z-score to ``[0, 100]`` using a logistic squashing.

    ``scale`` controls the slope: a smaller value makes the squashing
    more aggressive.  With the default ``scale=1`` a z-score of ``2``
    MAD-standard deviations maps to roughly 76, ``3`` to roughly 90,
    ``4`` to roughly 96.  Negative z-scores produce values below 50 —
    for one-sided signals (``direction="upper"`` / ``"lower"``) they
    are clipped at 0 before being passed in, so the sub-index sits in
    ``[50, 100]`` and is then re-centred to ``[0, 100]``.
    """

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
    """Time-aware exponential smoothing with a half-life in calendar days.

    The TZ-driven sub-indices and the final LSI are noisy because each
    of the underlying sources publishes at its own (irregular) cadence
    — repo auctions twice a week, OFZ auctions weekly, treasury data
    monthly.  Forward-filling those values onto a daily calendar
    creates step-like artefacts.  An EWMA with a half-life of ~1 week
    visibly smooths the daily series without introducing any look-
    ahead (the kernel is strictly causal).
    """

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
    """Rolling empirical-CDF percentile rank in ``[0, 100]``.

    For each date ``t`` returns the percentage of observations in
    ``(t - window_days, t]`` that are *strictly below* the current
    value (i.e. an empirical CDF lookup).  This produces a stationary,
    bounded series suitable for the final LSI calibration: a rank of
    95 means "today is more stressful than 95% of the last
    ``window_days`` days".
    """

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
    """Causal rolling winsorisation at the upper quantile ``upper_q``.

    Replaces any value that exceeds the rolling ``upper_q``-quantile in
    ``(t - window_days, t]`` with that quantile.  Uses only past data,
    so there is no look-ahead.  Driven by EDA findings: a handful of
    raw inputs (M3 OFZ ``cover_ratio``, ``demand_volume_mlnrub``;
    M1 ``spread_blnrub``) have extreme right tails (skew > 5,
    kurtosis > 50, p99/p50 > 10) where one-off events would otherwise
    dominate the rolling MAD window for years afterwards.
    """

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
    """Reindex a sparse series onto a daily calendar with ``ffill``.

    ``method='ffill'`` forward-fills the last known value (suitable for
    monthly / weekly data such as reserves or OFZ auctions).  Pass
    ``fill_value=0`` to encode "no event today".
    """

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
    """Robust online CUSUM detector with cooldown ("MIO" feature).

    Provided by the user.  Strictly causal: at each step ``t`` the
    rolling median / MAD are computed on history ``[t - window_size, t)``
    (i.e. *before* observing ``current_value``).  The output is in
    ``[0, 1]`` so it can be fed into the NSVM aggregator as a
    stationary feature without re-scaling.

    Parameters
    ----------
    window_size:
        Rolling window length in *observations* (the TZ asks for
        ~3 years ≈ 756 trading days).
    threshold:
        Maximum accumulated stress used to normalise the output to
        ``[0, 1]``.  A higher threshold makes the detector less
        sensitive.
    drift:
        Slack term (in robust z-units).  CUSUM only grows when the
        z-score exceeds ``drift`` — gates out small fluctuations.
    cooldown_factor:
        How quickly the accumulator decays when the signal returns to
        / below the median.  ``0.0`` instantly resets, ``1.0`` keeps
        the accumulator forever (pure CUSUM).  Default ``0.5`` halves
        on each calm step.
    """

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
    """Run :class:`RobustOnlineCUSUM` over a pandas Series.

    NaNs are passed through (output 0 for missing observations) so
    that masked / not-yet-published days do not poison the detector.
    """

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
