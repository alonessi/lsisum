"""Configuration for the RU Liquidity Sentinel pipeline.

All paths, weights, thresholds and date ranges are defined here so that
the rest of the pipeline stays declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


# Raw data files (CSVs are produced by the parsing layer that the user
# already implemented; this pipeline consumes them as is).
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


# MAD rolling window (in calendar days) per the specification.
MAD_WINDOW_DAYS = 365 * 1  # 1 years


# Thresholds for status colouring (TZ).
LSI_THRESHOLDS = {
    "green_max": 40.0,
    "yellow_max": 70.0,
}


# Backtest periods (from the TZ).
BACKTEST_EPISODES: List[Tuple[str, str, str]] = [
    ("dec_2014", "2014-12-01", "2015-01-31"),
    ("feb_mar_2022", "2022-02-15", "2022-04-15"),
    ("aug_2023", "2023-08-01", "2023-09-30"),
]


@dataclass
class PipelineConfig:
    """Top-level configuration for a pipeline run."""

    data_dir: Path = DATA_RAW
    artifacts_dir: Path = ARTIFACTS_DIR
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR

    mad_window_days: int = MAD_WINDOW_DAYS
    lsi_thresholds: Dict[str, float] = field(default_factory=lambda: dict(LSI_THRESHOLDS))
    backtest_episodes: List[Tuple[str, str, str]] = field(
        default_factory=lambda: list(BACKTEST_EPISODES)
    )

    # Whether to fit the NSVM aggregator. The LSI is derived from
    # its predictions (log-calibrated) and SHAP attribution, so it must
    # be trained for the pipeline to run end-to-end.
    fit_models: bool = True
    random_state: int = 42

    # Time-aware (walk-forward) cross-validation for the model suite.
    # n_splits feeds ``sklearn.model_selection.TimeSeriesSplit`` on the
    # train window (everything strictly before ``model_test_split_date``).
    # ``tune_models`` enables a small chronological grid search over
    # regularisation hyper-parameters; the chosen ``n_estimators`` for
    # GBMs is the median of the per-fold best_iteration found via early
    # stopping on the validation fold.
    model_test_split_date: str = "2024-01-01"
    cv_n_splits: int = 5
    tune_models: bool = True
    early_stopping_rounds: int = 50
    n_estimators_max: int = 1500

    # Auto-detection of stress episodes (used to evaluate the model on
    # *unknown* crises beyond the three TZ episodes).
    episode_min_days: int = 5
    episode_merge_gap_days: int = 5
    episode_top_k: int = 25

    # Sensitivity perturbation amplitude (±20%) — applied to NSVM
    # hyper-parameters (learning_rate, num_leaves, min_child_samples)
    # since the LSI no longer uses module weights.
    sensitivity_pct: float = 0.20

    def ensure_dirs(self) -> None:
        for path in (
            self.artifacts_dir,
            self.reports_dir,
            self.figures_dir,
            self.tables_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
