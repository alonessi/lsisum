from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.modules.m2_repo import build_m2
from ru_liquidity_sentinel.modules.m3_ofz import build_m3
from ru_liquidity_sentinel.modules.m5_treasury import build_m5


def test_m2_repo_cover_ratio_uses_consistent_units():
    calendar = pd.date_range("2024-01-01", periods=80, freq="D", name="date")
    repo = pd.DataFrame(
        {
            "date": [calendar[40]],
            "term_days": [7.0],
            "allotment_mlnrub": [1_000_000.0],
            "weighted_avg_rate_pct": [16.0],
            "bliq_repo_auction_vol": [2_500.0],
        }
    )
    keyrate = pd.DataFrame({"date": calendar, "key_rate_pct": [16.0] * len(calendar)})

    m2 = build_m2(repo, keyrate, calendar, mad_window_days=365)

    assert m2.loc[calendar[40], "m2_cover_ratio"] == 2.5
    assert m2.loc[calendar[40], "m2_flag_high_demand"] == 1


def test_m3_oversubscription_is_not_stress_flag():
    calendar = pd.date_range("2024-01-01", periods=80, freq="D", name="date")
    ofz = pd.DataFrame(
        {
            "date": [calendar[40]],
            "offer_volume_mlnrub": [100_000.0],
            "demand_volume_mlnrub": [300_000.0],
            "allotment_volume_mlnrub": [100_000.0],
            "weighted_avg_yield_pct": [12.0],
            "cover_ratio": [3.0],
        }
    )

    m3 = build_m3(ofz, calendar, mad_window_days=365)

    assert m3.loc[calendar[40], "m3_flag_undersubscribed"] == 0
    assert m3.loc[calendar[40], "m3_flag_oversubscribed"] == 0


def test_m5_mio_reacts_to_drain_not_inflow():
    calendar = pd.date_range("2024-01-01", periods=80, freq="D", name="date")
    bl = pd.DataFrame(
        {
            "date": calendar,
            "bank_corraccounts_blnrub": [5000.0] * len(calendar),
            "delta_1d_blnrub": [0.0] * len(calendar),
            "delta_5d_blnrub": [0.0] * 50 + [-2000.0] * 5 + [2500.0] * 25,
            "delta_22d_blnrub": [0.0] * len(calendar),
            "flag_budget_drain": [0] * len(calendar),
            "flag_budget_drain_strong": [0] * len(calendar),
        }
    )
    rk = pd.DataFrame({"date": calendar, "n_documents": [0] * len(calendar)})

    m5 = build_m5(bl, rk, calendar, mad_window_days=365)

    assert m5["m5_mio_cusum"].iloc[54] > 0.0
    assert m5["m5_mio_cusum"].iloc[-1] == 0.0


def test_tax_week_discount_uses_actual_feature_name():
    from ru_liquidity_sentinel.aggregate import apply_tax_week_discount

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    s = pd.Series([100.0, 100.0, 100.0], index=idx)
    features = pd.DataFrame({"m4_flag_tax_week": [0, 1, 0]}, index=idx)

    out = apply_tax_week_discount(s, features)

    assert out.tolist() == [100.0, 80.0, 100.0]
