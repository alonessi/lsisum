"""Configuration for the RU Liquidity Sentinel NSVM workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

RAW_FILES: Dict[str, str] = {
    "m1_rreserves": "m1_rreserves.csv",
    "m1_ruonia": "m1_ruonia.csv",
    "m2_keyrate": "m2_keyrate.csv",
    "m2_repo_auctions": "m2_repo_auctions.csv",
    "m3_ofz_auctions": "m3_ofz_auctions.csv",
    "m4_tax_calendar": "m4_tax_calendar.csv",
    "m4_tax_flags_daily": "m4_tax_flags_daily.csv",
    "m5_bliquidity": "m5_bliquidity.csv",
    "m5_roskazna_eks_deposits_index": "m5_roskazna_eks_deposits_index.csv",
    "m5_sors_funds_all": "m5_sors_funds_all.csv",
}

MAD_WINDOW_DAYS = 365

LSI_THRESHOLDS = {
    "green_max": 40.0,
    "yellow_max": 70.0,
}

BACKTEST_EPISODES: List[Tuple[str, str, str]] = [
    ("dec_2014", "2014-12-01", "2015-01-31"),
    ("feb_mar_2022", "2022-02-15", "2022-04-15"),
    ("aug_2023", "2023-08-01", "2023-09-30"),
]


@dataclass
class PipelineConfig:
    """Top-level configuration for data, NSVM and dashboard artifacts."""

    data_dir: Path = DATA_RAW
    artifacts_dir: Path = ARTIFACTS_DIR

    mad_window_days: int = MAD_WINDOW_DAYS
    lsi_thresholds: Dict[str, float] = field(default_factory=lambda: dict(LSI_THRESHOLDS))
    backtest_episodes: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(BACKTEST_EPISODES)
    )

    fit_models: bool = True
    random_state: int = 42

    model_test_split_date: str = "2024-01-01"
    cv_n_splits: int = 5
    tune_models: bool = True
    early_stopping_rounds: int = 50
    n_estimators_max: int = 1500

    nsvm_init_cutoff_date: str = "2025-01-01"
    nsvm_lsi_q_low: float = 0.20
    nsvm_lsi_q_high: float = 0.995
    nsvm_regime_nll_multiplier: float = 3.0
    nsvm_partial_fit_epochs: int = 2
    nsvm_partial_fit_lr: float = 1e-5

    episode_min_days: int = 5
    episode_merge_gap_days: int = 5
    episode_top_k: int = 25

    sensitivity_pct: float = 0.20

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
