"""Load parsed CSV datasets into feature-ready ``pandas`` DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_FILES


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected raw CSV not found: {path}")
    return pd.read_csv(path)


def load_rreserves(data_dir: Path) -> pd.DataFrame:
    """Monthly reserve-averaging series (M1)."""

    df = _read_csv(data_dir / RAW_FILES["m1_rreserves"])
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df = df.dropna(subset=["period_start"]).sort_values("period_start")
    df = df.rename(columns={"period_start": "date"})
    return df.reset_index(drop=True)


def load_ruonia(data_dir: Path) -> pd.DataFrame:
    """Daily RUONIA rate (M1)."""

    df = _read_csv(data_dir / RAW_FILES["m1_ruonia"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "ruonia_rate_pct", "volume_blnrub"]].reset_index(drop=True)


def load_keyrate(data_dir: Path) -> pd.DataFrame:
    """Daily key rate (M2 input)."""

    df = _read_csv(data_dir / RAW_FILES["m2_keyrate"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "key_rate_pct"]].reset_index(drop=True)


def load_repo_auctions(data_dir: Path) -> pd.DataFrame:
    """Bank of Russia repo auctions (M2)."""

    df = _read_csv(data_dir / RAW_FILES["m2_repo_auctions"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def load_ofz_auctions(data_dir: Path) -> pd.DataFrame:
    """Primary OFZ auctions (M3)."""

    df = _read_csv(data_dir / RAW_FILES["m3_ofz_auctions"])
    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce")
    df = df.dropna(subset=["auction_date"]).sort_values("auction_date")
    df = df.rename(columns={"auction_date": "date"})
    return df.reset_index(drop=True)


def load_tax_calendar(data_dir: Path) -> pd.DataFrame:
    """Tax calendar events (M4)."""

    df = _read_csv(data_dir / RAW_FILES["m4_tax_calendar"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def load_tax_flags_daily(data_dir: Path) -> pd.DataFrame:
    """Daily tax flags (M4)."""

    df = _read_csv(data_dir / RAW_FILES["m4_tax_flags_daily"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def load_bliquidity(data_dir: Path) -> pd.DataFrame:
    """Bank-sector liquidity indicators (M5)."""

    df = _read_csv(data_dir / RAW_FILES["m5_bliquidity"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def load_roskazna_index(data_dir: Path) -> pd.DataFrame:
    """Treasury deposit-allocation document index (M5)."""

    df = _read_csv(data_dir / RAW_FILES["m5_roskazna_eks_deposits_index"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date", "n_documents"]].reset_index(drop=True)


def load_sors_funds(data_dir: Path) -> pd.DataFrame:
    """Aggregated banking-sector funds breakdown (M5 supplementary)."""

    df = _read_csv(data_dir / RAW_FILES["m5_sors_funds_all"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["indicator"] = df["indicator"].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every configured raw dataset."""

    return {
        "rreserves": load_rreserves(data_dir),
        "ruonia": load_ruonia(data_dir),
        "keyrate": load_keyrate(data_dir),
        "repo_auctions": load_repo_auctions(data_dir),
        "ofz_auctions": load_ofz_auctions(data_dir),
        "tax_calendar": load_tax_calendar(data_dir),
        "tax_flags_daily": load_tax_flags_daily(data_dir),
        "bliquidity": load_bliquidity(data_dir),
        "roskazna_index": load_roskazna_index(data_dir),
        "sors_funds": load_sors_funds(data_dir),
    }
