"""Aggregation layer — Liquidity Stress Index (LSI).

This is the final layer that turns the per-module mad_scores + flags
+ MIO CUSUM features into the LSI. Design decisions:

* **Smoothing & Post-processing**: The raw signal is penalized by a factor of 0.8
  during known tax periods (module M4 `Tax_Week_Flag`) to account for expected
  predictability, followed by Exponential Moving Average (EMA) smoothing to reduce
  daily noise.
* **NSVMRegressor is the sole aggregator.** Two-headed LSTM predicting (mu, log_var).
  SHAP attribution via GradientExplainer, summed over the seq_len axis.
* **Raw → LSI via log-linear calibration** with two anchors q10 / q99 on train-set
  predictions. The proxy stress target is approximately log-normal.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from .nsvm_model import NSVMRegressor

from .config import (
    LSI_THRESHOLDS,
    MAD_WINDOW_DAYS,
)
from .normalize import rolling_mad_zscore, zscore_to_subindex


# ---------------------------------------------------------------------------
# Module-score helpers
# ---------------------------------------------------------------------------

MODULE_SCORE_FORMULA: Dict[str, Dict[str, float]] = {
    "M1": {
        "mad_weight": 0.7,
        "flag_weight": 6.0,
        "mio_weight": 25.0,
    },
    "M2": {
        "mad_weight": 0.7,
        "flag_weight": 8.0,
        "mio_weight": 25.0,
    },
    "M3": {
        "mad_weight": 0.7,
        "flag_weight": 10.0,
        "mio_weight": 25.0,
    },
    "M4": {
        "mad_weight": 0.5,
        "flag_weight": 4.0,
        "mio_weight": 15.0,
    },
    "M5": {
        "mad_weight": 0.7,
        "flag_weight": 7.0,
        "mio_weight": 25.0,
    },
}


def module_columns(features: pd.DataFrame, module: str) -> Dict[str, List[str]]:
    prefix = module.lower() + "_"
    cols = [c for c in features.columns if c.startswith(prefix)]
    out = {
        "mad": [c for c in cols if c.startswith(prefix + "mad")],
        "flag": [c for c in cols if c.startswith(prefix + "flag")],
        "mio": [c for c in cols if c.endswith("_mio_cusum")],
    }
    return out


def module_score(features: pd.DataFrame, module: str) -> pd.Series:
    cols = module_columns(features, module)
    if not cols["mad"] and not cols["flag"] and not cols["mio"]:
        return pd.Series(0.0, index=features.index, name=f"score_{module}")

    weights = MODULE_SCORE_FORMULA.get(module.upper(), MODULE_SCORE_FORMULA["M1"])

    mad_block = (
        features[cols["mad"]].mean(axis=1).fillna(0.0)
        if cols["mad"]
        else pd.Series(0.0, index=features.index)
    )
    mad_subindex = zscore_to_subindex(mad_block, scale=1.0)

    flag_block = (
        features[cols["flag"]].astype(float).sum(axis=1).fillna(0.0)
        if cols["flag"]
        else pd.Series(0.0, index=features.index)
    )
    mio_block = (
        features[cols["mio"]].mean(axis=1).fillna(0.0)
        if cols["mio"]
        else pd.Series(0.0, index=features.index)
    )

    score = (
        weights["mad_weight"] * mad_subindex
        + weights["flag_weight"] * flag_block
        + weights["mio_weight"] * mio_block
    ).clip(lower=0.0, upper=100.0)
    score.name = f"score_{module}"
    return score


def all_module_scores(features: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [module_score(features, m) for m in ("M1", "M2", "M3", "M4", "M5")],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Post-processing Helpers (EMA & M4 Penalty)
# ---------------------------------------------------------------------------

def apply_m4_penalty(series: pd.Series, features: pd.DataFrame) -> pd.Series:
    """Reduce indicator by 20% (multiply by 0.8) during predictable tax weeks."""
    if "Tax_Week_Flag" in features.columns:
        tax_mask = features["Tax_Week_Flag"].reindex(series.index).fillna(0) == 1
        return series.where(~tax_mask, series * 0.8)
    return series

def exponential_smoothing(series: pd.Series, span: int) -> pd.Series:
    """Apply exponential moving average smoothing."""
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# LSI frame assembly
# ---------------------------------------------------------------------------

def build_lsi_frame(
    features: pd.DataFrame,
    lgb_attribution: pd.DataFrame,
    ema_span: int = 5,
) -> pd.DataFrame:
    """Build the dashboard-facing LSI frame from the NSVM attribution."""

    out = pd.DataFrame(index=features.index)
    src = lgb_attribution["dsm_anomaly"] if "dsm_anomaly" in lgb_attribution.columns else lgb_attribution.iloc[:, 0]
    lsi_raw = src.reindex(features.index).fillna(0.0)

    # 1. Apply M4 Penalty to LSI and all contributions (to maintain sum)
    out["lsi"] = apply_m4_penalty(lsi_raw, features)

    for module in ("M1", "M2", "M3", "M4", "M5"):
        contrib_col = f"contrib_{module}"
        share_col = f"share_{module}"
        if contrib_col in lgb_attribution.columns:
            c_raw = lgb_attribution[contrib_col].reindex(features.index).fillna(0.0)
            out[contrib_col] = apply_m4_penalty(c_raw, features)
            out[share_col] = lgb_attribution[share_col].reindex(features.index).fillna(0.0)
        else:
            out[contrib_col] = 0.0
            out[share_col] = 0.0

    # 2. Apply EMA Smoothing
    if ema_span is not None and ema_span > 0:
        out["lsi"] = exponential_smoothing(out["lsi"], ema_span)
        for module in ("M1", "M2", "M3", "M4", "M5"):
            contrib_col = f"contrib_{module}"
            out[contrib_col] = exponential_smoothing(out[contrib_col], ema_span)

    # 3. Recalculate shares and scores to maintain mathematical consistency after smoothing
    for module in ("M1", "M2", "M3", "M4", "M5"):
        share_col = f"share_{module}"
        contrib_col = f"contrib_{module}"
        # Protect against division by zero
        lsi_nz = out["lsi"].replace(0.0, np.nan)
        out[share_col] = (out[contrib_col] / lsi_nz).fillna(0.0)
        out[f"score_{module}"] = (out[share_col] * 100.0).clip(0.0, 100.0)

    out["status"] = lsi_status(out["lsi"])
    return out


def lsi_status(lsi: pd.Series) -> pd.Series:
    bins = [
        -np.inf,
        LSI_THRESHOLDS["green_max"],
        LSI_THRESHOLDS["yellow_max"],
        np.inf,
    ]
    labels = ["green", "yellow", "red"]
    return pd.cut(lsi, bins=bins, labels=labels).astype(str)


# ---------------------------------------------------------------------------
# Proxy stress target
# ---------------------------------------------------------------------------

def build_proxy_target(
    features: pd.DataFrame,
    bliquidity: pd.DataFrame,
    ruonia: pd.DataFrame,
    keyrate: pd.DataFrame,
    mad_window_days: int,
) -> pd.Series:
    """Build a 0–100 proxy stress target from the CBR ground truth."""

    calendar = features.index
    bl = bliquidity.set_index("date").sort_index()
    ru = ruonia.set_index("date").sort_index()["ruonia_rate_pct"]
    kr = keyrate.set_index("date").sort_index()["key_rate_pct"]

    ru_d = ru.reindex(calendar).ffill()
    kr_d = kr.reindex(calendar).ffill()
    deficit = bl["deficit_total_blnrub"].reindex(calendar).ffill()

    rate_spread = (ru_d - kr_d).fillna(0.0)
    rate_z = rolling_mad_zscore(rate_spread, mad_window_days, direction="upper")
    deficit_z = rolling_mad_zscore(deficit.fillna(0.0), mad_window_days, direction="upper")

    combined_z = 0.6 * rate_z + 0.4 * deficit_z
    target = zscore_to_subindex(combined_z, scale=2.0).clip(0.0, 100.0)

    # M4 predictability correction
    target = apply_m4_penalty(target, features)

    return target.rename("proxy_stress")


# ---------------------------------------------------------------------------
# Feature selection for the aggregator models
# ---------------------------------------------------------------------------

def aggregator_feature_columns(features: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for module in ("M1", "M2", "M3", "M4", "M5"):
        groups = module_columns(features, module)
        cols.extend(groups["mad"])
        cols.extend(groups["flag"])
        cols.extend(groups["mio"])

    if "m4_seasonal_factor" in features.columns and "m4_seasonal_factor" not in cols:
        cols.append("m4_seasonal_factor")
    if "Tax_Week_Flag" in features.columns and "Tax_Week_Flag" not in cols:
        cols.append("Tax_Week_Flag")

    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c in seen or c not in features.columns:
            continue
        seen.add(c)
        out.append(c)
    return out


_FEATURE_TO_MODULE = {
    "m1_": "M1",
    "m2_": "M2",
    "m3_": "M3",
    "m4_": "M4",
    "m5_": "M5",
}

def _feature_to_module(feature: str) -> str:
    for prefix, module in _FEATURE_TO_MODULE.items():
        if feature.startswith(prefix):
            return module
    return "OTHER"


# ---------------------------------------------------------------------------
# Multi-model aggregation layer (NSVM)
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    feature_cols: List[str]
    train_idx: pd.Index
    test_idx: pd.Index
    model: Optional[Any] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    predictions: pd.Series = field(default_factory=pd.Series)
    feature_importance: Optional[pd.Series] = None
    module_attribution: Optional[pd.DataFrame] = None
    best_params: Optional[Dict[str, Any]] = None
    best_iteration: Optional[int] = None
    cv_results: Optional[pd.DataFrame] = None
    cv_summary: Optional[Dict[str, float]] = None


@dataclass
class MultiModelResult:
    feature_cols: List[str]
    target_name: str
    train_idx: pd.Index
    test_idx: pd.Index
    models: Dict[str, ModelResult] = field(default_factory=dict)

    def predictions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: m.predictions for name, m in self.models.items()}
        )

    def metrics_table(self) -> pd.DataFrame:
        rows = []
        for name, m in self.models.items():
            for split, metrics in (("train", m.train_metrics), ("test", m.test_metrics)):
                if not metrics:
                    continue
                row = {"model": name, "split": split}
                row.update(metrics)
                rows.append(row)
            if m.cv_summary:
                row = {"model": name, "split": "cv_val_mean"}
                row.update(m.cv_summary)
                rows.append(row)
        return pd.DataFrame(rows)

    def cv_table(self) -> pd.DataFrame:
        frames = []
        for name, m in self.models.items():
            if m.cv_results is None or m.cv_results.empty:
                continue
            df = m.cv_results.copy()
            df.insert(0, "model", name)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def best_params_table(self) -> pd.DataFrame:
        rows = []
        for name, m in self.models.items():
            if m.best_params is None:
                continue
            row = {"model": name}
            row.update({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in m.best_params.items()})
            if m.best_iteration is not None:
                row["best_iteration"] = m.best_iteration
            rows.append(row)
        return pd.DataFrame(rows)


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, threshold: float = None) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if threshold is None:
        threshold = float(np.nanpercentile(y_true.values, 90))
    y_bin = (y_true.values >= threshold).astype(int)
    if len(np.unique(y_bin)) < 2:
        return {}
    out: Dict[str, float] = {}
    out["roc_auc"] = float(roc_auc_score(y_bin, y_pred.values))
    out["pr_auc"] = float(average_precision_score(y_bin, y_pred.values))
    pred_thr = float(np.nanpercentile(y_pred.values, 90))
    pred_bin = (y_pred.values >= pred_thr).astype(int)
    if pred_bin.sum() > 0 and y_bin.sum() > 0:
        out["precision_p90"] = float(precision_score(y_bin, pred_bin))
        out["recall_p90"] = float(recall_score(y_bin, pred_bin))
        out["f1_p90"] = float(f1_score(y_bin, pred_bin))
    return out


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    err = (y_pred.values - y_true.values)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(y_true))}


def _expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values_iter = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, vs)) for vs in values_iter]


def _ts_cv_grid_search(
    factory: Callable[[Dict[str, Any]], Any],
    param_grid: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    fit_kwargs_for_es: Optional[Callable[[Any, pd.DataFrame, pd.Series], Dict[str, Any]]] = None,
    extract_best_iteration: Optional[Callable[[Any], Optional[int]]] = None,
    complexity_fn: Optional[Callable[[Dict[str, Any], Optional[int]], float]] = None,
) -> Tuple[Dict[str, Any], Optional[int], pd.DataFrame, Dict[str, float]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = list(tscv.split(X))

    rows: List[Dict[str, Any]] = []
    summary: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for config_idx, params in enumerate(param_grid):
        config_key = tuple(sorted(params.items()))
        fold_train_mae: List[float] = []
        fold_val_mae: List[float] = []
        fold_best_iters: List[int] = []
        for fold_id, (tr_pos, va_pos) in enumerate(folds):
            X_tr = X.iloc[tr_pos]
            y_tr = y.iloc[tr_pos]
            X_va = X.iloc[va_pos]
            y_va = y.iloc[va_pos]

            est = factory(params)
            fit_kwargs = (
                fit_kwargs_for_es(est, X_va, y_va) if fit_kwargs_for_es else {}
            )
            est.fit(X_tr, y_tr, **fit_kwargs)

            tr_pred = est.predict(X_tr)
            va_pred = est.predict(X_va)
            tr_mae = float(mean_absolute_error(y_tr, tr_pred))
            va_mae = float(mean_absolute_error(y_va, va_pred))
            best_iter = (
                extract_best_iteration(est) if extract_best_iteration else None
            )

            fold_train_mae.append(tr_mae)
            fold_val_mae.append(va_mae)
            if best_iter is not None and best_iter > 0:
                fold_best_iters.append(int(best_iter))

            rows.append(
                {
                    "config_idx": config_idx,
                    "params": json.dumps(params, sort_keys=True, default=str),
                    "fold": fold_id,
                    "n_train": int(len(tr_pos)),
                    "n_val": int(len(va_pos)),
                    "train_start": X.index[tr_pos[0]].date().isoformat(),
                    "train_end": X.index[tr_pos[-1]].date().isoformat(),
                    "val_start": X.index[va_pos[0]].date().isoformat(),
                    "val_end": X.index[va_pos[-1]].date().isoformat(),
                    "train_mae": tr_mae,
                    "val_mae": va_mae,
                    "best_iteration": best_iter,
                }
            )
        summary[config_key] = {
            "params": dict(params),
            "mean_train_mae": float(np.mean(fold_train_mae)),
            "mean_val_mae": float(np.mean(fold_val_mae)),
            "std_val_mae": float(np.std(fold_val_mae)),
            "median_best_iter": (
                int(np.median(fold_best_iters)) if fold_best_iters else None
            ),
        }

    raw_best_key = min(summary, key=lambda k: summary[k]["mean_val_mae"])
    raw_best = summary[raw_best_key]
    n_folds_eff = max(int(n_splits), 1)
    se = raw_best["std_val_mae"] / np.sqrt(n_folds_eff)
    threshold = raw_best["mean_val_mae"] + se
    candidates = [k for k, v in summary.items() if v["mean_val_mae"] <= threshold]
    if complexity_fn is not None and len(candidates) > 1:
        best_key = min(
            candidates,
            key=lambda k: (
                complexity_fn(summary[k]["params"], summary[k]["median_best_iter"]),
                summary[k]["mean_val_mae"],
            ),
        )
    else:
        best_key = raw_best_key
    best = summary[best_key]
    cv_results = pd.DataFrame(rows)
    cv_summary = {
        "cv_train_mae": best["mean_train_mae"],
        "cv_val_mae": best["mean_val_mae"],
        "cv_val_mae_std": best["std_val_mae"],
        "cv_train_val_gap": best["mean_train_mae"] - best["mean_val_mae"],
        "cv_n_candidates_within_1se": int(len(candidates)),
    }
    return best["params"], best["median_best_iter"], cv_results, cv_summary


def fit_model_suite(
        features: pd.DataFrame,
        target: pd.Series,
        feature_cols: Optional[List[str]] = None,
        test_split_date: str | None = "2024-01-01",
        random_state: int = 42,
        cv_n_splits: int = 5,
        tune_models: bool = True,
        early_stopping_rounds: int = 50,
        n_estimators_max: int = 1500,
        ema_span: int = 5,
) -> MultiModelResult:
    feature_cols = feature_cols or aggregator_feature_columns(features)
    df = features[feature_cols].join(target.rename("y")).dropna(subset=["y"])
    df = df.sort_index()
    X = df[feature_cols].fillna(0.0)
    y = df["y"]

    if test_split_date is None:
        train_mask = pd.Series(True, index=df.index)
    else:
        train_mask = df.index < pd.Timestamp(test_split_date)
    test_mask = ~train_mask
    if train_mask.sum() < 50 or test_mask.sum() < 10:
        train_mask = pd.Series(True, index=df.index)
        test_mask = ~train_mask

    train_idx = X.loc[train_mask].index
    test_idx = X.loc[test_mask].index

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    suite = MultiModelResult(
        feature_cols=feature_cols,
        target_name=str(target.name or "y"),
        train_idx=train_idx,
        test_idx=test_idx,
    )

    def _eval_and_store(
            name: str,
            model_obj: Any,
            is_gbm: bool = False,
            **kwargs: Any,
    ) -> None:
        pred_tr_raw = pd.Series(model_obj.predict(X_train), index=train_idx)
        pred_te_raw = (
            pd.Series(model_obj.predict(X_test), index=test_idx)
            if not X_test.empty
            else pd.Series(dtype=float)
        )
        raw_full = pd.concat([pred_tr_raw, pred_te_raw]).reindex(features.index)

        train_metrics = _regression_metrics(y_train, pred_tr_raw)
        test_metrics = (
            _regression_metrics(y_test, pred_te_raw) if not X_test.empty else {}
        )
        train_metrics.update(_classification_metrics(y_train, pred_tr_raw))
        if test_metrics:
            test_metrics.update(_classification_metrics(y_test, pred_te_raw))

        if is_gbm:
            # Передаем NSVM в gbm_lsi для SHAP GradientExplainer и лог-калибровки
            from .gbm_lsi import gbm_lsi as _gbm_lsi
            feature_to_module_gbm = {c: _feature_to_module(c) for c in feature_cols}
            lsi_series, gbm_attr, _cal = _gbm_lsi(
                model=model_obj,
                X_full=X,
                train_idx=train_idx,
                feature_to_module=feature_to_module_gbm,
            )

            # Post-process: Apply M4 Penalty and EMA
            lsi_series = apply_m4_penalty(lsi_series, features)
            if ema_span is not None and ema_span > 0:
                lsi_series = exponential_smoothing(lsi_series, ema_span)

            pred_full = lsi_series
            kwargs.pop("module_attribution", None)
            kwargs["module_attribution"] = gbm_attr
        else:
            pred_full = raw_full

        suite.models[name] = ModelResult(
            name=name,
            model=model_obj,
            feature_cols=feature_cols,
            train_idx=train_idx,
            test_idx=test_idx,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            predictions=pred_full,
            **kwargs,
        )

    # =====================================================================
    # NSVM aggregator
    # =====================================================================

    print("Инициализация и обучение Neural Stochastic Volatility Model (NSVM)...")

    # Гиперпараметры можно вынести в конфиг, пока захардкожены для ясности
    seq_length = 14
    hidden_dimensions = 64
    num_epochs = 30

    nsvm_model = NSVMRegressor(
        seq_len=seq_length,
        hidden_dim=hidden_dimensions,
        epochs=num_epochs,
        lr=0.001,
        device="cpu"  # Если окружение поддерживает GPU, можно поменять на "cuda"
    )

    # Обучаем нейросеть
    nsvm_model.fit(X_train, y_train)

    # Поскольку нейросети не имеют встроенного feature_importances_,
    # как деревья, мы создаем равномерную заглушку (dummy), чтобы
    # пайплайн не упал при сохранении метаданных.
    # Реальная важность признаков будет посчитана ниже через SHAP DeepExplainer.
    dummy_importance = pd.Series(1.0 / len(feature_cols), index=feature_cols)

    # Сохраняем модель в общий suite под именем "nsvm"
    # is_gbm=True критически важен: он запускает SHAP и логистическую калибровку
    _eval_and_store(
        "nsvm",
        nsvm_model,
        is_gbm=True,
        feature_importance=dummy_importance,
        best_params={"seq_len": seq_length, "hidden_dim": hidden_dimensions, "epochs": num_epochs},
        best_iteration=num_epochs,
    )

    return suite

def sensitivity_analysis(
    features: pd.DataFrame,
    base_lsi: pd.Series,
    feature_cols: List[str],
    proxy_target: pd.Series,
    pct: float = 0.20,
    test_split_date: str | None = "2024-01-01",
    random_state: int = 42,
    base_lgb_params: Dict[str, Any] | None = None,
    n_estimators: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity analysis через пертурбации гиперпараметров NSVM.

    Перебирает ±pct изменения learning_rate, hidden_dim и seq_len,
    обучает новую NSVM для каждого сценария и считает отклонение LSI
    от базовой линии.
    """
    from .nsvm_model import NSVMRegressor
    from .gbm_lsi import gbm_lsi as _gbm_lsi

    df = features[feature_cols].dropna(how="all").fillna(0.0)
    if test_split_date is None:
        train_mask = pd.Series(True, index=df.index)
    else:
        train_mask = df.index < pd.Timestamp(test_split_date)
    if train_mask.sum() < 50:
        train_mask = pd.Series(True, index=df.index)

    train_idx = df.index[train_mask]
    X_train = df.loc[train_idx]
    y_train = proxy_target.reindex(train_idx).fillna(0.0)

    feature_to_module = {c: _feature_to_module(c) for c in feature_cols}

    # Базовые гиперпараметры NSVM
    base_params = {
        "lr": 0.001,
        "hidden_dim": 64,
        "seq_len": 14,
    }

    rows = []
    columns: Dict[str, pd.Series] = {"lsi_base": base_lsi.reindex(features.index)}

    scenarios = [
        ("lr",         +1, f"lr_+{int(pct*100)}pct"),
        ("lr",         -1, f"lr_-{int(pct*100)}pct"),
        ("hidden_dim", +1, f"hidden_dim_+{int(pct*100)}pct"),
        ("hidden_dim", -1, f"hidden_dim_-{int(pct*100)}pct"),
        ("seq_len",    +1, f"seq_len_+{int(pct*100)}pct"),
        ("seq_len",    -1, f"seq_len_-{int(pct*100)}pct"),
    ]

    for param, sign, label in scenarios:
        perturbed = dict(base_params)
        new_val = perturbed[param] * (1.0 + sign * pct)
        if param in ("hidden_dim", "seq_len"):
            new_val = max(2, int(round(new_val)))
        perturbed[param] = new_val

        try:
            model = NSVMRegressor(
                seq_len=int(perturbed["seq_len"]),
                hidden_dim=int(perturbed["hidden_dim"]),
                epochs=15,
                lr=float(perturbed["lr"]),
                device="cpu",
            )
            model.fit(X_train, y_train)

            # Только калибровка без SHAP — атрибуция модулей здесь не нужна
            from .gbm_lsi import LogCalibration
            raw_full = pd.Series(model.predict(df), index=df.index)
            raw_train = raw_full.loc[raw_full.index.intersection(train_idx)].values
            cal = LogCalibration.fit(raw_train)
            lsi_arr = cal.transform(raw_full.values)
            lsi_series = pd.Series(lsi_arr, index=df.index, name="gbm_lsi")

            lsi_series = apply_m4_penalty(lsi_series, features)
            lsi_series = exponential_smoothing(lsi_series, span=5)

        except Exception:
            continue

    return pd.concat(columns, axis=1), pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Auto-detection of stress episodes
# ---------------------------------------------------------------------------

def detect_stress_episodes(
    lsi: pd.Series,
    threshold: float | None = None,
    min_episode_days: int = 5,
    merge_gap_days: int = 5,
    top_k: int | None = None,
) -> pd.DataFrame:
    s = lsi.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(
            columns=["start", "end", "length_days", "max_lsi", "mean_lsi", "rank"]
        )

    if threshold is None:
        threshold = float(np.nanpercentile(s.values, 90))

    above = s >= threshold
    if not above.any():
        return pd.DataFrame(
            columns=["start", "end", "length_days", "max_lsi", "mean_lsi", "rank"]
        )

    dates = s.index[above]
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = dates[0]
    cur_end = dates[0]
    for ts in dates[1:]:
        if (ts - cur_end).days <= merge_gap_days:
            cur_end = ts
        else:
            episodes.append((cur_start, cur_end))
            cur_start = ts
            cur_end = ts
    episodes.append((cur_start, cur_end))

    rows = []
    for start, end in episodes:
        window = s.loc[start:end]
        length = int((end - start).days) + 1
        if length < min_episode_days:
            continue
        rows.append(
            {
                "start": start,
                "end": end,
                "length_days": length,
                "max_lsi": float(window.max()),
                "mean_lsi": float(window.mean()),
            }
        )

    df = pd.DataFrame(rows).sort_values("max_lsi", ascending=False).reset_index(drop=True)
    if df.empty:
        return df.assign(rank=pd.Series(dtype=int))
    df["rank"] = np.arange(1, len(df) + 1)
    if top_k is not None:
        df = df.head(top_k)
    return df


# ---------------------------------------------------------------------------
# Noise decomposition for the NSVM-based LSI
# ---------------------------------------------------------------------------

def noise_breakdown(
    lsi_frame: pd.DataFrame,
) -> pd.DataFrame:
    contrib_cols = [
        f"contrib_{m}" for m in ("M1", "M2", "M3", "M4", "M5")
        if f"contrib_{m}" in lsi_frame.columns
    ]
    if not contrib_cols:
        return pd.DataFrame(columns=["term", "component", "contribution", "share"])

    contribs = lsi_frame[contrib_cols].diff().dropna()

    rows = []
    for col in contrib_cols:
        module = col.replace("contrib_", "")
        var = float(contribs[col].var(ddof=0))
        rows.append(
            {
                "term": module,
                "component": "var",
                "contribution": var,
            }
        )

    modules = [c.replace("contrib_", "") for c in contrib_cols]
    for i, mi in enumerate(modules):
        for mj in modules[i + 1 :]:
            cov = float(
                contribs[f"contrib_{mi}"].cov(contribs[f"contrib_{mj}"], ddof=0)
            )
            rows.append(
                {
                    "term": f"{mi}×{mj}",
                    "component": "cov",
                    "contribution": 2.0 * cov,
                }
            )

    df = pd.DataFrame(rows)
    total = df["contribution"].sum()
    df["share"] = df["contribution"] / total if total != 0.0 else 0.0
    df = df.sort_values("contribution", ascending=False).reset_index(drop=True)
    return df


__all__ = [
    "MODULE_SCORE_FORMULA",
    "module_columns",
    "module_score",
    "all_module_scores",
    "apply_m4_penalty",
    "exponential_smoothing",
    "build_lsi_frame",
    "lsi_status",
    "build_proxy_target",
    "aggregator_feature_columns",
    "fit_model_suite",
    "ModelResult",
    "MultiModelResult",
    "sensitivity_analysis",
    "detect_stress_episodes",
    "noise_breakdown",
]