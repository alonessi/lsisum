from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import loaders
from .aggregate import (
    _feature_to_module,
    aggregator_feature_columns,
    apply_m4_penalty,
    build_lsi_frame,
    detect_stress_episodes,
    noise_breakdown,
)
from .backtest import evaluate_episodes
from .config import PipelineConfig
from .gbm_lsi import ECDFCalibration, shap_module_attribution
from .nsvm_model import NSVMAnomalyDetector
from .pipeline import build_features, seed_everything


HISTORICAL_LSI_NAME = "historical_lsi.parquet"
NSVM_WEIGHTS_NAME = "nsvm_weights.pt"
CALIBRATOR_NAME = "calibrator.pkl"
INCREMENTAL_META_NAME = "incremental_metadata.json"
RAW_SCORE_COL = "raw_nsvm_score"
RAW_SCORE_EMA_COL = "raw_nsvm_score_ema"
CALIBRATED_SIGNAL_COL = "nsvm_calibrated_signal"


@dataclass
class IncrementalUpdateResult:
    last_saved_date: pd.Timestamp | None
    last_available_date: pd.Timestamp
    new_rows: int
    mean_nll: float | None
    threshold: float | None
    regime_shift: bool
    partial_fit_ran: bool
    historical_path: Path


def _paths(cfg: PipelineConfig) -> dict[str, Path]:
    root = Path(cfg.artifacts_dir)
    return {
        "historical_lsi": root / HISTORICAL_LSI_NAME,
        "weights": root / NSVM_WEIGHTS_NAME,
        "calibrator": root / CALIBRATOR_NAME,
        "metadata": root / INCREMENTAL_META_NAME,
    }


def _feature_to_module_map(feature_cols: list[str]) -> dict[str, str]:
    return {col: _feature_to_module(col) for col in feature_cols}


def _save_calibrator(calibrator: ECDFCalibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(calibrator, fh)


def _load_calibrator(path: Path) -> ECDFCalibration:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _write_dashboard_artifacts(
    cfg: PipelineConfig,
    features: pd.DataFrame,
    historical_lsi: pd.DataFrame,
    metadata: dict,
) -> None:
    root = Path(cfg.artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)

    features.to_parquet(root / "features_daily.parquet")
    historical_lsi.to_parquet(root / "lsi_daily.parquet")
    historical_lsi[["lsi"]].rename(columns={"lsi": "nsvm"}).to_parquet(
        root / "model_predictions.parquet"
    )

    auto_episodes = detect_stress_episodes(
        historical_lsi["lsi"],
        threshold=float(np.nanpercentile(historical_lsi["lsi"].dropna().values, 90)),
        min_episode_days=cfg.episode_min_days,
        top_k=cfg.episode_top_k,
    )
    backtest_df = evaluate_episodes(historical_lsi["lsi"], cfg.backtest_episodes)
    nb = noise_breakdown(historical_lsi)

    auto_episodes.to_csv(root / "auto_episodes.csv", index=False)
    backtest_df.to_csv(root / "backtest_episodes.csv", index=False)
    nb.to_csv(root / "noise_breakdown.csv", index=False)
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")

    attr_cols = [
        "dsm_anomaly",
        *[f"share_{m}" for m in ("M1", "M2", "M3", "M4", "M5")],
        *[f"contrib_{m}" for m in ("M1", "M2", "M3", "M4", "M5")],
    ]
    attr_cols = [c for c in attr_cols if c in historical_lsi.columns]
    if attr_cols:
        historical_lsi[attr_cols].to_parquet(root / "model_nsvm_module_attribution.parquet")

    pd.DataFrame(
        [
            {
                "model": "nsvm",
                "split": "incremental_state",
                "mean_nll": metadata.get("train_mean_nll"),
                "std_nll": metadata.get("train_std_nll"),
                "max_nll": metadata.get("train_max_nll"),
                "n": metadata.get("train_n"),
            }
        ]
    ).to_csv(root / "model_metrics.csv", index=False)


def _add_internal_score_columns(
    lsi_frame: pd.DataFrame,
    raw_scores: pd.Series,
    raw_scores_ema: pd.Series,
    calibrated_signal: pd.Series,
) -> pd.DataFrame:
    out = lsi_frame.copy()
    out[RAW_SCORE_COL] = raw_scores.reindex(out.index).astype(float)
    out[RAW_SCORE_EMA_COL] = raw_scores_ema.reindex(out.index).astype(float)
    out[CALIBRATED_SIGNAL_COL] = calibrated_signal.reindex(out.index).astype(float)
    return out


def _metadata(
    cfg: PipelineConfig,
    features: pd.DataFrame,
    lsi: pd.DataFrame,
    feature_cols: list[str],
    train_scores: pd.Series,
    cutoff_date: str,
) -> dict:
    return {
        "n_days": int(features.shape[0]),
        "date_min": features.index.min().date().isoformat(),
        "date_max": features.index.max().date().isoformat(),
        "lsi_method": "nsvm_ecdf_incremental",
        "nsvm_init_cutoff_date": cutoff_date,
        "feature_cols": feature_cols,
        "train_mean_nll": float(train_scores.mean()),
        "train_std_nll": float(train_scores.std(ddof=0)),
        "train_max_nll": float(train_scores.max()),
        "train_n": int(train_scores.shape[0]),
        "regime_threshold": float(train_scores.mean() + cfg.nsvm_regime_nll_multiplier * train_scores.std(ddof=0)),
        "lsi_quantiles": {
            "q50": float(lsi["lsi"].median()),
            "q90": float(lsi["lsi"].quantile(0.9)),
            "q99": float(lsi["lsi"].quantile(0.99)),
        },
    }


def _build_full_lsi(
    cfg: PipelineConfig,
    features: pd.DataFrame,
    model: NSVMAnomalyDetector,
    calibrator: ECDFCalibration,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_full = features[feature_cols].fillna(0.0)
    raw_scores = pd.Series(model.predict(X_full), index=X_full.index, name=RAW_SCORE_COL)
    seq_len = int(getattr(model, "seq_len", 14))
    if len(raw_scores) > seq_len:
        raw_scores.iloc[:seq_len] = np.nan
    raw_scores_ema = raw_scores.ewm(span=7, adjust=False).mean()
    calibrated = pd.Series(
        calibrator.transform(raw_scores_ema.bfill().ffill().values),
        index=X_full.index,
        name=CALIBRATED_SIGNAL_COL,
    )

    attr = shap_module_attribution(model, X_full, _feature_to_module_map(feature_cols), calibrated)
    lsi = build_lsi_frame(features, attr)
    lsi = _add_internal_score_columns(lsi, raw_scores, raw_scores_ema, calibrated)
    return lsi, raw_scores, raw_scores_ema, calibrated


def offline_initialize(cfg: PipelineConfig | None = None, cutoff_date: str | None = None) -> IncrementalUpdateResult:
    cfg = cfg or PipelineConfig()
    cfg.ensure_dirs()
    seed_everything(cfg.random_state)
    cutoff_date = cutoff_date or cfg.nsvm_init_cutoff_date
    paths = _paths(cfg)

    data = loaders.load_all(cfg.data_dir)
    features = build_features(data, cfg)
    feature_cols = aggregator_feature_columns(features)
    X = features[feature_cols].dropna(how="all").fillna(0.0).sort_index()
    train_mask = X.index < pd.Timestamp(cutoff_date)
    if train_mask.sum() < 50:
        raise RuntimeError(f"Too few rows before {cutoff_date} to train NSVM")

    model = NSVMAnomalyDetector(seq_len=14, hidden_dim=64, epochs=30, lr=0.001, device="cpu")
    model.fit(X.loc[train_mask])

    raw_train = pd.Series(model.predict(X.loc[train_mask]), index=X.loc[train_mask].index)
    seq_len = int(getattr(model, "seq_len", 14))
    if len(raw_train) > seq_len:
        raw_train.iloc[:seq_len] = np.nan
    raw_train_ema = raw_train.ewm(span=7, adjust=False).mean().dropna()
    calibrator = ECDFCalibration.fit(raw_train_ema.values)

    historical_lsi, _raw, _raw_ema, _cal = _build_full_lsi(cfg, features, model, calibrator, feature_cols)
    metadata = _metadata(cfg, features, historical_lsi, feature_cols, raw_train_ema, cutoff_date)

    historical_lsi.to_parquet(paths["historical_lsi"])
    model.save_weights(paths["weights"], feature_cols=feature_cols, extra=metadata)
    _save_calibrator(calibrator, paths["calibrator"])
    paths["metadata"].write_text(json.dumps(metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_dashboard_artifacts(cfg, features, historical_lsi, metadata)

    return IncrementalUpdateResult(
        last_saved_date=None,
        last_available_date=features.index.max(),
        new_rows=int(features.shape[0]),
        mean_nll=None,
        threshold=metadata["regime_threshold"],
        regime_shift=False,
        partial_fit_ran=False,
        historical_path=paths["historical_lsi"],
    )


def fetch_fresh_data(cfg: PipelineConfig, date_from: pd.Timestamp, date_to: pd.Timestamp | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parsers_root = repo_root / "parsers"
    fetch_all = parsers_root / "scripts" / "fetch_all.py"
    processed = parsers_root / "data" / "processed"
    raw = Path(cfg.data_dir)
    if not fetch_all.exists():
        raise FileNotFoundError(f"Parser entry point not found: {fetch_all}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(parsers_root), str(repo_root / "src"), env.get("PYTHONPATH", "")])
    cmd = [
        sys.executable,
        str(fetch_all),
        "--from",
        date_from.date().isoformat(),
    ]
    if date_to is not None:
        cmd.extend(["--to", date_to.date().isoformat()])

    result = subprocess.run(cmd, cwd=str(parsers_root), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Fresh-data parsers exited with code {result.returncode}")

    raw.mkdir(parents=True, exist_ok=True)
    for csv in processed.glob("*.csv"):
        _merge_processed_csv(csv, raw / csv.name)


def _merge_processed_csv(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy2(src, dst)
        return

    old = pd.read_csv(dst)
    new = pd.read_csv(src)
    if old.empty:
        new.to_csv(dst, index=False)
        return
    if new.empty:
        return

    combined = pd.concat([old, new], ignore_index=True)
    date_cols = [c for c in ("date", "period_start", "auction_date") if c in combined.columns]
    for col in date_cols:
        combined[col] = pd.to_datetime(combined[col], errors="coerce")

    subset = list(combined.columns)
    combined = combined.drop_duplicates(subset=subset, keep="last")
    if date_cols:
        combined = combined.sort_values(date_cols[0])
        for col in date_cols:
            combined[col] = combined[col].dt.date.astype(str)
    combined.to_csv(dst, index=False)


def _ema_continue(values: pd.Series, previous_ema: float | None, span: int) -> pd.Series:
    alpha = 2.0 / (span + 1.0)
    out = []
    prev = previous_ema
    for value in values.astype(float):
        if not np.isfinite(value):
            out.append(prev if prev is not None else np.nan)
            continue
        prev = float(value) if prev is None or not np.isfinite(prev) else alpha * float(value) + (1.0 - alpha) * prev
        out.append(prev)
    return pd.Series(out, index=values.index)


def _finalize_new_lsi_rows(
    cfg: PipelineConfig,
    features_new: pd.DataFrame,
    attr_new: pd.DataFrame,
    historical_lsi: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(index=features_new.index)
    lsi_raw = attr_new["dsm_anomaly"].reindex(out.index).fillna(0.0)
    out["lsi"] = apply_m4_penalty(lsi_raw, features_new)

    for module in ("M1", "M2", "M3", "M4", "M5"):
        contrib_col = f"contrib_{module}"
        share_col = f"share_{module}"
        out[contrib_col] = apply_m4_penalty(attr_new[contrib_col].reindex(out.index).fillna(0.0), features_new)
        out[share_col] = attr_new[share_col].reindex(out.index).fillna(0.0)

    out["lsi"] = _ema_continue(out["lsi"], float(historical_lsi["lsi"].iloc[-1]), span=5)
    for module in ("M1", "M2", "M3", "M4", "M5"):
        contrib_col = f"contrib_{module}"
        prev = float(historical_lsi[contrib_col].iloc[-1]) if contrib_col in historical_lsi.columns else 0.0
        out[contrib_col] = _ema_continue(out[contrib_col], prev, span=5)

    for module in ("M1", "M2", "M3", "M4", "M5"):
        share_col = f"share_{module}"
        contrib_col = f"contrib_{module}"
        lsi_nz = out["lsi"].replace(0.0, np.nan)
        out[share_col] = (out[contrib_col] / lsi_nz).fillna(0.0)
        out[f"score_{module}"] = (out[share_col] * 100.0).clip(0.0, 100.0)

    from .aggregate import lsi_status

    out["status"] = lsi_status(out["lsi"])
    return out


def incremental_update(
    cfg: PipelineConfig | None = None,
    fetch: bool = True,
    today: pd.Timestamp | None = None,
) -> IncrementalUpdateResult:
    cfg = cfg or PipelineConfig()
    cfg.ensure_dirs()
    seed_everything(cfg.random_state)
    paths = _paths(cfg)

    if not paths["historical_lsi"].exists() or not paths["weights"].exists() or not paths["calibrator"].exists():
        return offline_initialize(cfg)

    if paths["metadata"].exists():
        existing_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if existing_metadata.get("nsvm_init_cutoff_date") != cfg.nsvm_init_cutoff_date:
            return offline_initialize(cfg)

    historical_lsi = pd.read_parquet(paths["historical_lsi"]).sort_index()
    historical_lsi.index = pd.to_datetime(historical_lsi.index)
    last_saved_date = pd.Timestamp(historical_lsi.index.max()).normalize()
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()

    if fetch and last_saved_date < today:
        fetch_fresh_data(cfg, last_saved_date + pd.Timedelta(days=1), today)

    data = loaders.load_all(cfg.data_dir)
    features = build_features(data, cfg)
    features.index = pd.to_datetime(features.index)
    last_available_date = pd.Timestamp(features.index.max()).normalize()
    new_idx = features.index[features.index > last_saved_date]

    if len(new_idx) == 0:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8")) if paths["metadata"].exists() else {}
        return IncrementalUpdateResult(
            last_saved_date=last_saved_date,
            last_available_date=last_available_date,
            new_rows=0,
            mean_nll=None,
            threshold=metadata.get("regime_threshold"),
            regime_shift=False,
            partial_fit_ran=False,
            historical_path=paths["historical_lsi"],
        )

    model = NSVMAnomalyDetector.load_weights(paths["weights"], device="cpu")
    calibrator = _load_calibrator(paths["calibrator"])
    feature_cols = list(getattr(model, "feature_cols_", []) or model.extra_.get("feature_cols") or [])
    if not feature_cols:
        feature_cols = aggregator_feature_columns(features)

    X_all = features[feature_cols].fillna(0.0)
    X_new = X_all.loc[new_idx]
    context = X_all.loc[X_all.index <= last_saved_date].tail(model.seq_len)

    raw_new = pd.Series(model.predict(X_new, context=context), index=X_new.index, name=RAW_SCORE_COL)
    prev_raw_ema = None
    if RAW_SCORE_EMA_COL in historical_lsi.columns and pd.notna(historical_lsi[RAW_SCORE_EMA_COL].iloc[-1]):
        prev_raw_ema = float(historical_lsi[RAW_SCORE_EMA_COL].iloc[-1])
    raw_new_ema = _ema_continue(raw_new, prev_raw_ema, span=7).rename(RAW_SCORE_EMA_COL)
    calibrated_new = pd.Series(
        calibrator.transform(raw_new_ema.bfill().ffill().values),
        index=X_new.index,
        name=CALIBRATED_SIGNAL_COL,
    )

    attr_new = shap_module_attribution(model, X_new, _feature_to_module_map(feature_cols), calibrated_new)
    lsi_new = _finalize_new_lsi_rows(cfg, features.loc[new_idx], attr_new, historical_lsi)
    lsi_new = _add_internal_score_columns(lsi_new, raw_new, raw_new_ema, calibrated_new)

    combined = pd.concat([historical_lsi.loc[historical_lsi.index <= last_saved_date], lsi_new], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8")) if paths["metadata"].exists() else {}
    train_mean = float(metadata.get("train_mean_nll", raw_new.mean()))
    train_std = float(metadata.get("train_std_nll", raw_new.std(ddof=0) if len(raw_new) > 1 else 0.0))
    threshold = float(metadata.get("regime_threshold", train_mean + cfg.nsvm_regime_nll_multiplier * train_std))
    mean_nll = float(raw_new.mean())
    regime_shift = bool(mean_nll > threshold)
    partial_fit_ran = False

    if regime_shift:
        model.partial_fit(
            X_new,
            context=context,
            epochs=cfg.nsvm_partial_fit_epochs,
            lr=cfg.nsvm_partial_fit_lr,
        )
        model.save_weights(paths["weights"], feature_cols=feature_cols, extra=metadata)
        partial_fit_ran = True

    metadata.update(
        {
            "date_max": combined.index.max().date().isoformat(),
            "n_days": int(combined.shape[0]),
            "last_incremental_update": pd.Timestamp.now().isoformat(),
            "last_incremental_rows": int(len(new_idx)),
            "last_incremental_mean_nll": mean_nll,
            "last_incremental_threshold": threshold,
            "last_incremental_regime_shift": regime_shift,
            "last_partial_fit_ran": partial_fit_ran,
            "lsi_quantiles": {
                "q50": float(combined["lsi"].median()),
                "q90": float(combined["lsi"].quantile(0.9)),
                "q99": float(combined["lsi"].quantile(0.99)),
            },
        }
    )

    combined.to_parquet(paths["historical_lsi"])
    paths["metadata"].write_text(json.dumps(metadata, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_dashboard_artifacts(cfg, features.loc[: combined.index.max()], combined, metadata)

    return IncrementalUpdateResult(
        last_saved_date=last_saved_date,
        last_available_date=last_available_date,
        new_rows=int(len(new_idx)),
        mean_nll=mean_nll,
        threshold=threshold,
        regime_shift=regime_shift,
        partial_fit_ran=partial_fit_ran,
        historical_path=paths["historical_lsi"],
    )


__all__ = [
    "offline_initialize",
    "incremental_update",
    "fetch_fresh_data",
    "IncrementalUpdateResult",
    "HISTORICAL_LSI_NAME",
    "NSVM_WEIGHTS_NAME",
    "CALIBRATOR_NAME",
]
