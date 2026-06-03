from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.modules.m1_reserves import build_m1


def test_positive_reserve_spread_is_not_shortfall_stress():
    calendar = pd.date_range("2023-01-01", periods=520, freq="D", name="date")
    period_starts = pd.date_range("2023-01-01", periods=18, freq="30D")
    spreads = [50.0] * 17 + [350.0]
    rreserves = pd.DataFrame(
        {
            "date": period_starts,
            "actual_balances_blnrub": [1050.0 + s for s in spreads],
            "required_avg_blnrub": [1050.0] * len(spreads),
            "spread_blnrub": spreads,
        }
    )
    ruonia = pd.DataFrame({"date": calendar, "ruonia_rate_pct": [10.0] * len(calendar)})
    keyrate = pd.DataFrame({"date": calendar, "key_rate_pct": [10.0] * len(calendar)})

    m1 = build_m1(rreserves, ruonia, keyrate, calendar, mad_window_days=365)

    assert m1["m1_spread_blnrub"].iloc[-1] > 0.0
    assert m1["m1_reserve_shortfall_blnrub"].iloc[-1] < 0.0
    assert m1["m1_mad_spread"].iloc[-1] == 0.0
    assert m1["m1_mio_cusum"].iloc[-1] == 0.0
