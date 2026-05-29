"""End-to-end pipeline orchestration.

``run_pipeline`` performs the full computation:

1. Load all parsed CSVs.
2. Build a daily calendar covering every dataset.
3. Compute the per-module feature frames (M1..M5) — TZ outputs only:
   ``mad_score`` columns, binary flags, and a single MIO CUSUM
   accumulator per module.  No EWMA smoothing anywhere.
4. Build a model-free proxy stress target.
5. Fit the NSVM aggregator with walk-forward CV.
6. Build the dashboard LSI from the NSVM SHAP attribution
   (lsi = log-calibrated predictions; contrib_M{n} = share_M{n} × lsi
   where share_M{n} is normalised |SHAP| over module-specific
   features).
7. Decompose the noise (Var(ΔLSI)) into per-module + cross-covariance
   components.
8. Auto-detect stress episodes (top-K) for generalisation evaluation.
9. Run the TZ backtest, hold-out and sensitivity analyses (LSI
   robustness to ±20% perturbations of NSVM hyper-parameters).
10. Persist the resulting frames + metrics to ``artifacts/``.
"""

from __future__ import annotations

import os
import random
import torch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import loaders
from .aggregate import (
    MultiModelResult,
    aggregator_feature_columns,
    build_proxy_target,
    detect_stress_episodes,
    fit_model_suite,
    build_lsi_frame,
    noise_breakdown,
    sensitivity_analysis,
)
from .backtest import evaluate_episodes, holdout_metrics
from .config import PipelineConfig
from .modules import build_m1, build_m2, build_m3, build_m4, build_m5


def seed_everything(seed: int = 42):
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # На случай, если код будет запускаться на GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class PipelineRun:
    config: PipelineConfig
    features: pd.DataFrame
    lsi: pd.DataFrame
    proxy_target: pd.Series
    backtest: pd.DataFrame
    holdout: Dict[str, float]
    sensitivity: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    auto_episodes: pd.DataFrame
    noise_breakdown: pd.DataFrame
    models: Optional[MultiModelResult] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def save(self, root: Path | None = None) -> Dict[str, Path]:
        root = Path(root or self.config.artifacts_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        feat_path = root / "features_daily.parquet"
        lsi_path = root / "lsi_daily.parquet"
        target_path = root / "proxy_target.parquet"
        sens_path = root / "sensitivity.parquet"
        sens_sum_path = root / "sensitivity_summary.csv"
        backtest_path = root / "backtest_episodes.csv"
        auto_path = root / "auto_episodes.csv"
        noise_path = root / "noise_breakdown.csv"
        meta_path = root / "metadata.json"

        self.features.to_parquet(feat_path)
        self.lsi.to_parquet(lsi_path)
        self.proxy_target.to_frame("proxy_stress").to_parquet(target_path)
        self.sensitivity.to_parquet(sens_path)
        self.sensitivity_summary.to_csv(sens_sum_path, index=False)
        self.backtest.to_csv(backtest_path, index=False)
        self.auto_episodes.to_csv(auto_path, index=False)
        self.noise_breakdown.to_csv(noise_path, index=False)

        meta = dict(self.metadata)
        meta["holdout"] = self.holdout
        if self.models is not None:
            meta["models"] = {}
            preds = self.models.predictions_frame()
            preds.to_parquet(root / "model_predictions.parquet")
            self.models.metrics_table().to_csv(
                root / "model_metrics.csv", index=False
            )
            cv_table = self.models.cv_table()
            if not cv_table.empty:
                cv_table.to_csv(root / "model_cv_results.csv", index=False)
            best_table = self.models.best_params_table()
            if not best_table.empty:
                best_table.to_csv(root / "model_best_params.csv", index=False)
            for name, m in self.models.models.items():
                entry = {
                    "train_metrics": m.train_metrics,
                    "test_metrics": m.test_metrics,
                }
                if m.cv_summary:
                    entry["cv_summary"] = m.cv_summary
                if m.best_params:
                    entry["best_params"] = m.best_params
                if m.best_iteration is not None:
                    entry["best_iteration"] = m.best_iteration
                if m.feature_importance is not None:
                    entry["feature_importance"] = m.feature_importance.to_dict()
                if m.module_attribution is not None:
                    m.module_attribution.to_parquet(
                        root / f"model_{name}_module_attribution.parquet"
                    )
                meta["models"][name] = entry
        meta_path.write_text(
            json.dumps(meta, default=str, ensure_ascii=False, indent=2)
        )

        paths.update(
            {
                "features": feat_path,
                "lsi": lsi_path,
                "proxy_target": target_path,
                "sensitivity": sens_path,
                "sensitivity_summary": sens_sum_path,
                "backtest": backtest_path,
                "auto_episodes": auto_path,
                "noise_breakdown": noise_path,
                "metadata": meta_path,
            }
        )
        return paths


def _build_calendar(data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Daily calendar covering every input dataset."""

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

    start = min(candidates)
    end = max(candidates)
    return pd.date_range(start=start, end=end, freq="D", name="date")


def build_features(
    data: Dict[str, pd.DataFrame],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Daily feature matrix combining all five modules (TZ outputs)."""

    calendar = _build_calendar(data)

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


def run_pipeline(cfg: PipelineConfig | None = None) -> PipelineRun:
    """Полный цикл выполнения пайплайна с модулем адаптации и фиксом вклада модулей."""

    cfg = cfg or PipelineConfig()
    cfg.ensure_dirs()

    # 1. Загрузка и подготовка
    seed_everything(cfg.random_state)
    data = loaders.load_all(cfg.data_dir)
    features = build_features(data, cfg)

    proxy_target = build_proxy_target(
        features=features,
        bliquidity=data["bliquidity"],
        ruonia=data["ruonia"],
        keyrate=data["keyrate"],
        mad_window_days=cfg.mad_window_days,
    )

    # 2. Обучение NSVM
    models = fit_model_suite(
        features=features,
        target=proxy_target,
        test_split_date=cfg.model_test_split_date,
        random_state=cfg.random_state,
        cv_n_splits=cfg.cv_n_splits,
    )

    # ==========================================================
    # АДАПТАЦИЯ И ПЕРЕСЧЕТ В КЛАДА (FIX)
    # ==========================================================
    adaptive_model_update(models, features, proxy_target)

    from .aggregate import _feature_to_module
    from .gbm_lsi import gbm_lsi as _gbm_lsi

    nsvm_container = models.models["nsvm"]
    X_full_f = features[nsvm_container.feature_cols].fillna(0.0)
    f2m = {c: _feature_to_module(c) for c in X_full_f.columns}

    # nsvm_container.model — это NSVMRegressor (имеет _create_sequences и predict)
    _, new_attr, _ = _gbm_lsi(
        model=nsvm_container.model,
        X_full=X_full_f,
        train_idx=nsvm_container.train_idx,
        feature_to_module=f2m
    )

    # ВАЖНО: Добавляем префикс 'contrib_', который ищет фронтенд
    if 'M1' in new_attr.columns and 'contrib_M1' not in new_attr.columns:
        new_attr = new_attr.rename(columns={m: f"contrib_{m}" for m in ["M1", "M2", "M3", "M4", "M5"]})

    nsvm_container.module_attribution = new_attr
    # ==========================================================

    # 3. Расчет финального LSI (теперь вклад модулей будет внутри lsi)
    nsvm_attr = nsvm_container.module_attribution
    lsi = build_lsi_frame(features, nsvm_attr)

    # 4. Анализ эпизодов и гладкость
    auto_episodes = detect_stress_episodes(
        lsi["lsi"],
        threshold=float(np.nanpercentile(lsi["lsi"].dropna().values, 90)),
        min_episode_days=cfg.episode_min_days,
        top_k=cfg.episode_top_k,
    )

    backtest_df = evaluate_episodes(lsi["lsi"], cfg.backtest_episodes)
    nb = noise_breakdown(lsi)

    # 5. Сбор ПОЛНЫХ метаданных (чтобы reporting.py не падал)
    metadata = {
        "n_days": int(features.shape[0]),
        "date_min": features.index.min().date().isoformat(),
        "date_max": features.index.max().date().isoformat(),
        "lsi_method": "nsvm_shap_adaptive_log_calibrated",
        "n_auto_episodes": int(len(auto_episodes)),
        "lsi_smoothness": {
            "std_day_diff": float(lsi["lsi"].diff().std()),
        },
        "module_share_means": {
            m: float(lsi[f"contrib_{m}"].abs().mean())  # Берем средний вклад по модулям
            for m in ("M1", "M2", "M3", "M4", "M5")
        },
        "lsi_quantiles": {
            "q50": float(lsi["lsi"].median()),
            "q90": float(lsi["lsi"].quantile(0.9)),
            "q99": float(lsi["lsi"].quantile(0.99)),
        },
        "nsvm_best_params": dict(nsvm_container.best_params or {}),
        "nsvm_cv_summary": dict(nsvm_container.cv_summary or {}),
    }

    # 5. Hold-out метрики
    holdout = holdout_metrics(
        lsi=lsi["lsi"],
        proxy_target=proxy_target,
        holdout_start=cfg.model_test_split_date,
    )

    # 6. Sensitivity analysis — пертурбации гиперпараметров NSVM
    nsvm_res = models.models["nsvm"]
    sens_df, sens_summary = sensitivity_analysis(
        features=features,
        base_lsi=lsi["lsi"],
        feature_cols=nsvm_res.feature_cols,
        proxy_target=proxy_target,
        pct=cfg.sensitivity_pct,
        test_split_date=cfg.model_test_split_date,
        random_state=cfg.random_state,
    )
    run = PipelineRun(
        config=cfg,
        features=features,
        lsi=lsi,
        proxy_target=proxy_target,
        backtest=backtest_df,
        holdout=holdout,
        sensitivity=sens_df,
        sensitivity_summary=sens_summary,
        auto_episodes=auto_episodes,
        noise_breakdown=nb,
        models=models,
        metadata=metadata,
    )
    run.save()
    print("✅ Пайплайн завершен успешно. Отчеты сохранены.")
    return run


def adaptive_model_update(models, new_features, new_target):
    """
    Модуль адаптивного дообучения (Online Learning).
    Срабатывает только при реальных рыночных шоках (если текущая ошибка в 1.5 раза выше нормы).
    """
    res = models.models.get("nsvm")
    if res is None or res.model is None:
        return

    actual_model = res.model
    train_features = res.feature_cols

    # Берем последние 30 дней для замера ошибки
    X_adaptation = new_features[train_features].tail(30).fillna(0.0)
    y_adaptation = new_target.tail(30).fillna(0.0)

    current_preds = actual_model.predict(X_adaptation)
    current_mae = np.mean(np.abs(current_preds - y_adaptation.values))

    cv_mae = res.train_metrics.get("mae", 1.0)
    threshold = cv_mae * 1.5

    # =======================================================
    # БОЕВОЙ РЕЖИМ: Адаптация только при реальной разладке
    # =======================================================
    if current_mae > threshold:
        print(f"⚠️  ВНИМАНИЕ: Обнаружена разладка! Текущая MAE: {current_mae:.2f} (Порог: {threshold:.2f})")
        print("⏳ Запуск адаптивного дообучения (partial_fit) на новых данных...")

        try:
            actual_model.partial_fit(
                X_adaptation,
                y_adaptation,
                epochs=7,
                lr=1e-4
            )
            print("✅ Веса NSVM успешно обновлены. Система адаптирована к новым условиям.")
        except Exception as e:
            print(f"❌ Ошибка во время выполнения partial_fit: {e}")
    else:
        print(f"✅ Рынок стабилен. Текущая MAE: {current_mae:.2f} (Порог: {threshold:.2f}).")