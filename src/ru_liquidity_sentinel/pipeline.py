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
    detect_stress_episodes,
    fit_model_suite,
    build_lsi_frame,
    noise_breakdown,
    sensitivity_analysis,
)
from .backtest import evaluate_episodes
from .config import PipelineConfig
from .modules import build_m1, build_m2, build_m3, build_m4, build_m5


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    backtest: pd.DataFrame
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
        sens_path = root / "sensitivity.parquet"
        sens_sum_path = root / "sensitivity_summary.csv"
        backtest_path = root / "backtest_episodes.csv"
        auto_path = root / "auto_episodes.csv"
        noise_path = root / "noise_breakdown.csv"
        meta_path = root / "metadata.json"

        self.features.to_parquet(feat_path)
        self.lsi.to_parquet(lsi_path)
        self.sensitivity.to_parquet(sens_path)
        self.sensitivity_summary.to_csv(sens_sum_path, index=False)
        self.backtest.to_csv(backtest_path, index=False)
        self.auto_episodes.to_csv(auto_path, index=False)
        self.noise_breakdown.to_csv(noise_path, index=False)

        meta = dict(self.metadata)
        if self.models is not None:
            meta["models"] = {}
            preds = self.models.predictions_frame()
            preds.to_parquet(root / "model_predictions.parquet")
            self.models.metrics_table().to_csv(root / "model_metrics.csv", index=False)
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
                if m.cv_summary: entry["cv_summary"] = m.cv_summary
                if m.best_params: entry["best_params"] = m.best_params
                if m.best_iteration is not None: entry["best_iteration"] = m.best_iteration
                if m.feature_importance is not None: entry["feature_importance"] = m.feature_importance.to_dict()
                if m.module_attribution is not None:
                    m.module_attribution.to_parquet(root / f"model_{name}_module_attribution.parquet")
                meta["models"][name] = entry

        meta_path.write_text(json.dumps(meta, default=str, ensure_ascii=False, indent=2))
        paths.update({
            "features": feat_path, "lsi": lsi_path, "sensitivity": sens_path,
            "sensitivity_summary": sens_sum_path, "backtest": backtest_path,
            "auto_episodes": auto_path, "noise_breakdown": noise_path, "metadata": meta_path,
        })
        return paths


def _build_calendar(data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    candidates = []
    for key in ("ruonia", "keyrate", "bliquidity", "tax_flags_daily", "rreserves", "ofz_auctions", "repo_auctions", "sors_funds", "roskazna_index"):
        df = data.get(key)
        if df is None or df.empty or "date" not in df.columns: continue
        candidates.append(df["date"].min())
        candidates.append(df["date"].max())
    if not candidates: raise RuntimeError("No date columns available to build a calendar")
    return pd.date_range(start=min(candidates), end=max(candidates), freq="D", name="date")


def build_features(data: Dict[str, pd.DataFrame], cfg: PipelineConfig) -> pd.DataFrame:
    calendar = _build_calendar(data)
    m1 = build_m1(rreserves=data["rreserves"], ruonia=data["ruonia"], keyrate=data["keyrate"], calendar=calendar, mad_window_days=cfg.mad_window_days)
    m2 = build_m2(repo_auctions=data["repo_auctions"], keyrate=data["keyrate"], calendar=calendar, mad_window_days=cfg.mad_window_days)
    m3 = build_m3(ofz_auctions=data["ofz_auctions"], calendar=calendar, mad_window_days=cfg.mad_window_days)
    m4 = build_m4(tax_flags_daily=data["tax_flags_daily"], calendar=calendar, mad_window_days=cfg.mad_window_days)
    m5 = build_m5(bliquidity=data["bliquidity"], roskazna_index=data["roskazna_index"], calendar=calendar, mad_window_days=cfg.mad_window_days)
    features = pd.concat([m1, m2, m3, m4, m5], axis=1)
    features.index.name = "date"
    return features


def run_pipeline(cfg: PipelineConfig | None = None) -> PipelineRun:
    cfg = cfg or PipelineConfig()
    cfg.ensure_dirs()

    seed_everything(cfg.random_state)
    data = loaders.load_all(cfg.data_dir)
    features = build_features(data, cfg)

    models = fit_model_suite(
        features=features,
        test_split_date=cfg.model_test_split_date,
        random_state=cfg.random_state,
        cv_n_splits=cfg.cv_n_splits,
    )

    from .aggregate import _feature_to_module
    from .gbm_lsi import gbm_lsi as _gbm_lsi

    nsvm_container = models.models["nsvm"]
    X_full_f = features[nsvm_container.feature_cols].fillna(0.0)
    f2m = {c: _feature_to_module(c) for c in X_full_f.columns}

    # 1. СНАЧАЛА получаем честный исторический бэктест (базовой моделью)
    _, hist_attr, _ = _gbm_lsi(
        model=nsvm_container.model,
        X_full=X_full_f,
        train_idx=nsvm_container.train_idx,
        feature_to_module=f2m
    )

    # 2. ТЕПЕРЬ запускаем адаптацию на последних 30 днях (для прода)
    # Убедись, что внутри этой функции исправлен расчет порога (threshold) и lr=1e-5!
    adaptive_model_update(models, features)

    # 3. Если веса обновились, получаем предсказания адаптированной модели
    _, new_attr, _ = _gbm_lsi(
        model=nsvm_container.model,
        X_full=X_full_f,
        train_idx=nsvm_container.train_idx,
        feature_to_module=f2m
    )

    # 4. "Сшиваем" результаты: берем честную историю и заменяем только последние 30 дней
    # на результаты адаптированной модели
    final_attr = hist_attr.copy()
    tail_len = 30
    final_attr.iloc[-tail_len:] = new_attr.iloc[-tail_len:]

    if 'M1' in final_attr.columns and 'contrib_M1' not in final_attr.columns:
        final_attr = final_attr.rename(columns={m: f"contrib_{m}" for m in ["M1", "M2", "M3", "M4", "M5"]})

    nsvm_container.module_attribution = final_attr
    nsvm_attr = nsvm_container.module_attribution
    lsi = build_lsi_frame(features, nsvm_attr)

    auto_episodes = detect_stress_episodes(
        lsi["lsi"],
        threshold=float(np.nanpercentile(lsi["lsi"].dropna().values, 90)),
        min_episode_days=cfg.episode_min_days,
        top_k=cfg.episode_top_k,
    )

    backtest_df = evaluate_episodes(lsi["lsi"], cfg.backtest_episodes)
    nb = noise_breakdown(lsi)

    metadata = {
        "n_days": int(features.shape[0]),
        "date_min": features.index.min().date().isoformat(),
        "date_max": features.index.max().date().isoformat(),
        "lsi_method": "nsvm_shap_adaptive_log_calibrated",
        "n_auto_episodes": int(len(auto_episodes)),
        "lsi_smoothness": {"std_day_diff": float(lsi["lsi"].diff().std())},
        "module_share_means": {m: float(lsi[f"contrib_{m}"].abs().mean()) for m in ("M1", "M2", "M3", "M4", "M5")},
        "lsi_quantiles": {"q50": float(lsi["lsi"].median()), "q90": float(lsi["lsi"].quantile(0.9)), "q99": float(lsi["lsi"].quantile(0.99))},
        "nsvm_best_params": dict(nsvm_container.best_params or {}),
        "nsvm_cv_summary": dict(nsvm_container.cv_summary or {}),
    }

    nsvm_res = models.models["nsvm"]
    sens_df, sens_summary = sensitivity_analysis(
        features=features,
        base_lsi=lsi["lsi"],
        feature_cols=nsvm_res.feature_cols,
        pct=cfg.sensitivity_pct,
        test_split_date=cfg.model_test_split_date,
        random_state=cfg.random_state,
    )

    run = PipelineRun(
        config=cfg,
        features=features,
        lsi=lsi,
        backtest=backtest_df,
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


def adaptive_model_update(models, new_features):
    res = models.models.get("nsvm")
    if res is None or res.model is None:
        return

    actual_model = res.model
    train_features = res.feature_cols

    # ---> ИСПРАВЛЕНИЕ <---
    # Берем 30 дней + seq_len дней контекста!
    seq_len = getattr(actual_model, 'seq_len', 14)
    # Берем 30 дней для обучения + 14 дней контекста, чтобы LSTM было от чего оттолкнуться
    X_adaptation = new_features[train_features].tail(30 + seq_len).fillna(0.0)

    # А вот проверять порог (NLL) продолжаем строго на последних 30 днях
    X_check = new_features[train_features].tail(30).fillna(0.0)
    current_nll_scores = actual_model.predict(X_check)
    current_mean_nll = np.mean(current_nll_scores)

    cv_nll = res.train_metrics.get("mean_nll", -1.0)
    threshold = cv_nll + abs(cv_nll) * 0.5

    if current_mean_nll > threshold:
        print(f"⚠️ ВНИМАНИЕ: Смена рыночного режима! Текущий NLL: {current_mean_nll:.2f} (Порог: {threshold:.2f})")
        print("⏳ Запуск unsupervised дообучения (partial_fit)...")
        try:
            # Модель "проглотит" контекст и обновит веса только на честных данных
            actual_model.partial_fit(X_adaptation, epochs=2, lr=1e-5)
            print("✅ Веса NSVM обновлены.")
        except Exception as e:
            print(f"❌ Ошибка partial_fit: {e}")
    else:
        print(f"✅ Рынок в рамках распределения. Текущий NLL: {current_mean_nll:.2f} (Порог: {threshold:.2f}).")