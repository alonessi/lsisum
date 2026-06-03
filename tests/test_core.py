from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.aggregate import detect_stress_episodes, lsi_status, module_columns
from ru_liquidity_sentinel.normalize import (
    robust_online_cusum_series,
    rolling_mad_zscore,
    winsorize_rolling,
    zscore_to_subindex,
)
from ru_liquidity_sentinel.nsvm_lsi import ECDFCalibration


def calendar(n: int = 200) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_mad_zscore_constant_series_is_neutral():
    s = pd.Series(np.full(120, 7.0), index=calendar(120))
    z = rolling_mad_zscore(s, window_days=90, min_periods=20)
    assert (z == 0.0).all()


def test_subindex_bounds():
    sub = zscore_to_subindex(pd.Series([0.0, 1.0, 3.0, 10.0]))
    assert (sub >= 0.0).all()
    assert (sub <= 100.0).all()


def test_winsorize_rolling_caps_outlier_against_past_window():
    idx = calendar(120)
    s = pd.Series(np.ones(120), index=idx)
    s.iloc[-1] = 1000.0
    capped = winsorize_rolling(s, window_days=90, upper_q=0.99, min_periods=30)
    assert capped.iloc[-1] < 1000.0


def test_cusum_reacts_to_sustained_positive_shift():
    idx = calendar(120)
    s = pd.Series(np.r_[np.zeros(60), np.full(60, 5.0)], index=idx)
    out = robust_online_cusum_series(s, window_size=60, threshold=10.0)
    assert out.iloc[:21].eq(0.0).all()
    assert out.iloc[-20:].max() > 0.0
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_ecdf_calibrator_is_monotone_and_bounded():
    cal = ECDFCalibration.fit(np.arange(1000, dtype=float))
    out = cal.transform(np.array([-1.0, 0.0, 250.0, 500.0, 999.0, 1500.0]))
    assert np.diff(out).min() >= 0.0
    assert out.min() >= 0.0
    assert out.max() <= 100.0
    assert out[-1] == 100.0


def test_detect_stress_episodes_synthetic():
    idx = calendar(100)
    s = pd.Series(0.0, index=idx)
    s.iloc[10:15] = 80.0
    s.iloc[50:58] = 90.0
    episodes = detect_stress_episodes(s, threshold=70.0, min_episode_days=3)
    assert len(episodes) == 2
    assert episodes["max_lsi"].is_monotonic_decreasing


def test_lsi_status_thresholds():
    status = lsi_status(pd.Series([10.0, 50.0, 90.0]))
    assert status.tolist() == ["green", "yellow", "red"]


def test_module_columns_groups_features():
    df = pd.DataFrame(
        {
            "m1_mad_spread": [0.0],
            "m1_flag_end_of_period": [0],
            "m1_mio_cusum": [0.0],
            "m2_mad_cover": [0.0],
        }
    )
    cols = module_columns(df, "M1")
    assert cols["mad"] == ["m1_mad_spread"]
    assert cols["flag"] == ["m1_flag_end_of_period"]
    assert cols["mio"] == ["m1_mio_cusum"]


def test_import_nsvm_entry_points():
    pytest.importorskip("torch")
    from ru_liquidity_sentinel.nsvm_model import NSVMAnomalyDetector

    assert NSVMAnomalyDetector(seq_len=3).seq_len == 3
