"""M5 feature builder: treasury flows and bank-sector liquidity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize import (
    align_to_calendar,
    rolling_mad_zscore,
    robust_online_cusum_series,
)


_BL_COLUMN_MAP = [
    ("bank_corraccounts_blnrub", "m5_eks_balance_blnrub"),
    ("delta_1d_blnrub", "m5_eks_delta_1d"),
    ("delta_5d_blnrub", "m5_eks_delta_5d"),
    ("delta_22d_blnrub", "m5_eks_delta_22d"),
    ("flag_budget_drain", "m5_flag_budget_drain"),
    ("flag_budget_drain_strong", "m5_flag_budget_drain_strong"),
]


def build_m5(
    bliquidity: pd.DataFrame,
    roskazna_index: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    mad_window_days: int,
) -> pd.DataFrame:
    """Build daily M5 features from liquidity and treasury-flow data."""

    bl = bliquidity.copy().set_index("date").sort_index()

    out = pd.DataFrame(index=calendar)
    out.index.name = "date"

    for src, dst in _BL_COLUMN_MAP:
        if src in bl.columns:
            out[dst] = align_to_calendar(bl[src], calendar, method="ffill")
        else:
            out[dst] = np.nan

    out[["m5_flag_budget_drain", "m5_flag_budget_drain_strong"]] = out[
        ["m5_flag_budget_drain", "m5_flag_budget_drain_strong"]
    ].fillna(0).astype("int8")

    out["m5_mad_treasury_drain"] = rolling_mad_zscore(
        out["m5_eks_delta_5d"], window_days=mad_window_days, direction="lower"
    )

    rk = roskazna_index.copy().set_index("date").sort_index()
    rk_col = (
        rk["volume_placed_blnrub"]
        if "volume_placed_blnrub" in rk.columns
        else rk.get("n_documents", pd.Series(dtype=float))
    )
    rk_vol = align_to_calendar(rk_col, calendar, method="ffill")

    rk_delta_5d = rk_vol.diff(periods=5).fillna(0.0)
    out["m5_roskazna_delta_5d_blnrub"] = rk_delta_5d

    out["m5_mad_roskazna_drain"] = rolling_mad_zscore(
        rk_delta_5d, window_days=mad_window_days, direction="lower"
    )

    roskazna_drain_flag = (rk_delta_5d <= -300.0).astype("int8")
    out["m5_flag_budget_drain"] = (
        out["m5_flag_budget_drain"] | roskazna_drain_flag
    ).astype("int8")

    out["m5_mio_cusum"] = robust_online_cusum_series(
        (-out["m5_eks_delta_5d"]).rename("m5"),
        window_size=min(756, mad_window_days),
    ).values

    return out
