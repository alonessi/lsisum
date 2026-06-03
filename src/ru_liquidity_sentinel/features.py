from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .modules import build_m1, build_m2, build_m3, build_m4, build_m5


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and Torch."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_calendar(data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Build the daily calendar covering all loaded raw data sources."""

    candidates = []
    for key in (
        "ruonia",
        "keyrate",
        "bliquidity",
        "tax_flags_daily",
        "rreserves",
        "ofz_auctions",
        "repo_auctions",
        "sors_funds",
        "roskazna_index",
    ):
        df = data.get(key)
        if df is None or df.empty or "date" not in df.columns:
            continue
        candidates.append(df["date"].min())
        candidates.append(df["date"].max())
    if not candidates:
        raise RuntimeError("No date columns available to build a calendar")
    return pd.date_range(start=min(candidates), end=max(candidates), freq="D", name="date")


def build_features(data: Dict[str, pd.DataFrame], cfg: PipelineConfig) -> pd.DataFrame:
    """Build the full daily feature matrix from raw parser outputs."""

    calendar = build_calendar(data)
    m1 = build_m1(
        rreserves=data["rreserves"],
        ruonia=data["ruonia"],
        keyrate=data["keyrate"],
        calendar=calendar,
        mad_window_days=cfg.mad_window_days,
    )
    m2 = build_m2(
        repo_auctions=data["repo_auctions"],
        keyrate=data["keyrate"],
        calendar=calendar,
        mad_window_days=cfg.mad_window_days,
    )
    m3 = build_m3(
        ofz_auctions=data["ofz_auctions"],
        calendar=calendar,
        mad_window_days=cfg.mad_window_days,
    )
    m4 = build_m4(
        tax_flags_daily=data["tax_flags_daily"],
        calendar=calendar,
        mad_window_days=cfg.mad_window_days,
    )
    m5 = build_m5(
        bliquidity=data["bliquidity"],
        roskazna_index=data["roskazna_index"],
        calendar=calendar,
        mad_window_days=cfg.mad_window_days,
    )
    features = pd.concat([m1, m2, m3, m4, m5], axis=1)
    features.index.name = "date"
    return features


__all__ = ["seed_everything", "build_calendar", "build_features"]
