"""Aggregation layer for the NSVM Liquidity Stress Index."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from .config import LSI_THRESHOLDS
from .normalize import zscore_to_subindex

MODULE_SCORE_FORMULA: Dict[str, Dict[str, float]] = {
    "M1": {"mad_weight": 0.7, "flag_weight": 6.0, "mio_weight": 25.0},
    "M2": {"mad_weight": 0.7, "flag_weight": 8.0, "mio_weight": 25.0},
    "M3": {"mad_weight": 0.7, "flag_weight": 10.0, "mio_weight": 25.0},
    "M4": {"mad_weight": 0.5, "flag_weight": 4.0, "mio_weight": 15.0},
    "M5": {"mad_weight": 0.7, "flag_weight": 7.0, "mio_weight": 25.0},
}


def module_columns(features: pd.DataFrame, module: str) -> Dict[str, List[str]]:
    prefix = module.lower() + "_"
    cols = [c for c in features.columns if c.startswith(prefix)]
    return {
        "mad": [c for c in cols if c.startswith(prefix + "mad")],
        "flag": [c for c in cols if c.startswith(prefix + "flag")],
        "mio": [c for c in cols if c.endswith("_mio_cusum")],
    }


def module_score(features: pd.DataFrame, module: str) -> pd.Series:
    cols = module_columns(features, module)
    if not cols["mad"] and not cols["flag"] and not cols["mio"]:
        return pd.Series(0.0, index=features.index, name=f"score_{module}")

    weights = MODULE_SCORE_FORMULA.get(module.upper(), MODULE_SCORE_FORMULA["M1"])
    mad_block = features[cols["mad"]].mean(axis=1).fillna(0.0) if cols["mad"] else pd.Series(0.0, index=features.index)
    mad_subindex = zscore_to_subindex(mad_block, scale=1.0)
    flag_block = features[cols["flag"]].astype(float).sum(axis=1).fillna(0.0) if cols["flag"] else pd.Series(0.0,
                                                                                                             index=features.index)
    mio_block = features[cols["mio"]].mean(axis=1).fillna(0.0) if cols["mio"] else pd.Series(0.0, index=features.index)

    score = (weights["mad_weight"] * mad_subindex + weights["flag_weight"] * flag_block + weights[
        "mio_weight"] * mio_block).clip(lower=0.0, upper=100.0)
    score.name = f"score_{module}"
    return score


def all_module_scores(features: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([module_score(features, m) for m in ("M1", "M2", "M3", "M4", "M5")], axis=1)


def apply_m4_penalty(series: pd.Series, features: pd.DataFrame) -> pd.Series:
    """Reduce indicator by 20% (multiply by 0.8) during predictable tax weeks."""
    tax_col = "m4_flag_tax_week" if "m4_flag_tax_week" in features.columns else "Tax_Week_Flag"
    if tax_col in features.columns:
        tax_mask = features[tax_col].reindex(series.index).fillna(0) == 1
        return series.where(~tax_mask, series * 0.8)
    return series


def exponential_smoothing(series: pd.Series, span: int) -> pd.Series:
    """Apply exponential moving average smoothing."""
    return series.ewm(span=span, adjust=False).mean()


def build_lsi_frame(features: pd.DataFrame, nsvm_attribution: pd.DataFrame, ema_span: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    src = nsvm_attribution["dsm_anomaly"] if "dsm_anomaly" in nsvm_attribution.columns else nsvm_attribution.iloc[:, 0]
    lsi_raw = src.reindex(features.index).fillna(0.0)

    out["lsi"] = apply_m4_penalty(lsi_raw, features)

    for module in ("M1", "M2", "M3", "M4", "M5"):
        contrib_col = f"contrib_{module}"
        share_col = f"share_{module}"
        if contrib_col in nsvm_attribution.columns:
            c_raw = nsvm_attribution[contrib_col].reindex(features.index).fillna(0.0)
            out[contrib_col] = apply_m4_penalty(c_raw, features)
            out[share_col] = nsvm_attribution[share_col].reindex(features.index).fillna(0.0)
        else:
            out[contrib_col] = 0.0
            out[share_col] = 0.0

    if ema_span is not None and ema_span > 0:
        out["lsi"] = exponential_smoothing(out["lsi"], ema_span)
        for module in ("M1", "M2", "M3", "M4", "M5"):
            contrib_col = f"contrib_{module}"
            out[contrib_col] = exponential_smoothing(out[contrib_col], ema_span)

    for module in ("M1", "M2", "M3", "M4", "M5"):
        share_col = f"share_{module}"
        contrib_col = f"contrib_{module}"
        lsi_nz = out["lsi"].replace(0.0, np.nan)
        out[share_col] = (out[contrib_col] / lsi_nz).fillna(0.0)
        out[f"score_{module}"] = (out[share_col] * 100.0).clip(0.0, 100.0)

    out["status"] = lsi_status(out["lsi"])
    return out


def lsi_status(lsi: pd.Series) -> pd.Series:
    bins = [-np.inf, LSI_THRESHOLDS["green_max"], LSI_THRESHOLDS["yellow_max"], np.inf]
    labels = ["green", "yellow", "red"]
    return pd.cut(lsi, bins=bins, labels=labels).astype(str)


def aggregator_feature_columns(features: pd.DataFrame) -> List[str]:
    cols: List[str] = []

    for module in ("M1", "M2", "M3", "M5"):
        groups = module_columns(features, module)
        cols.extend(groups["mad"])
        cols.extend(groups["flag"])
        cols.extend(groups["mio"])

    if "m4_seasonal_factor" in features.columns and "m4_seasonal_factor" not in cols:
        cols.append("m4_seasonal_factor")

    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c in seen or c not in features.columns:
            continue
        seen.add(c)
        out.append(c)
    return out


_FEATURE_TO_MODULE = {"m1_": "M1", "m2_": "M2", "m3_": "M3", "m4_": "M4", "m5_": "M5"}


def _feature_to_module(feature: str) -> str:
    for prefix, module in _FEATURE_TO_MODULE.items():
        if feature.startswith(prefix):
            return module
    return "OTHER"

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
    train_idx: pd.Index
    test_idx: pd.Index
    models: Dict[str, ModelResult] = field(default_factory=dict)

    def predictions_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: m.predictions for name, m in self.models.items()})

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
            row.update({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in m.best_params.items()})
            if m.best_iteration is not None:
                row["best_iteration"] = m.best_iteration
            rows.append(row)
        return pd.DataFrame(rows)


def _unsupervised_metrics(nll_scores: pd.Series) -> Dict[str, float]:
    scores = nll_scores.values
    return {
        "mean_nll": float(np.mean(scores)),
        "std_nll": float(np.std(scores)),
        "max_nll": float(np.max(scores)),
        "n": int(len(scores))
    }


def fit_model_suite(
        features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        test_split_date: str | None = "2024-01-01",
        random_state: int = 42,
        cv_n_splits: int = 5,
        ema_span: int = 5,
) -> MultiModelResult:
    feature_cols = feature_cols or aggregator_feature_columns(features)
    df = features[feature_cols].dropna(how="all").sort_index()
    X = df[feature_cols].fillna(0.0)

    if test_split_date is None:
        train_mask = pd.Series(True, index=df.index)
    else:
        train_mask = df.index < pd.Timestamp(test_split_date)
    test_mask = ~train_mask
    if train_mask.sum() < 50:
        train_mask = pd.Series(True, index=df.index)
        test_mask = ~train_mask

    train_idx = X.loc[train_mask].index
    test_idx = X.loc[test_mask].index

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    suite = MultiModelResult(
        feature_cols=feature_cols,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    def _eval_and_store(name: str, model_obj: Any, **kwargs: Any) -> None:
        pred_tr_raw = pd.Series(model_obj.predict(X_train), index=train_idx)
        pred_te_raw = pd.Series(model_obj.predict(X_test), index=test_idx) if not X_test.empty else pd.Series(
            dtype=float)

        train_metrics = _unsupervised_metrics(pred_tr_raw)
        test_metrics = _unsupervised_metrics(pred_te_raw) if not X_test.empty else {}

        from .nsvm_lsi import nsvm_lsi as _nsvm_lsi
        feature_to_module_nsvm = {c: _feature_to_module(c) for c in feature_cols}
        lsi_series, nsvm_attr, _cal = _nsvm_lsi(
            model=model_obj,
            X_full=X,
            train_idx=train_idx,
            feature_to_module=feature_to_module_nsvm,
        )

        lsi_series = apply_m4_penalty(lsi_series, features)
        if ema_span is not None and ema_span > 0:
            lsi_series = exponential_smoothing(lsi_series, ema_span)

        kwargs["module_attribution"] = nsvm_attr

        suite.models[name] = ModelResult(
            name=name,
            model=model_obj,
            feature_cols=feature_cols,
            train_idx=train_idx,
            test_idx=test_idx,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            predictions=lsi_series,
            **kwargs,
        )

    print("Training unsupervised NSVM...")
    seq_length = 14
    hidden_dimensions = 64
    num_epochs = 30

    from .nsvm_model import NSVMAnomalyDetector
    nsvm_model = NSVMAnomalyDetector(
        seq_len=seq_length,
        hidden_dim=hidden_dimensions,
        epochs=num_epochs,
        lr=0.001,
        device="cpu"
    )

    nsvm_model.fit(X_train)
    dummy_importance = pd.Series(1.0 / len(feature_cols), index=feature_cols)

    _eval_and_store(
        "nsvm",
        nsvm_model,
        feature_importance=dummy_importance,
        best_params={"seq_len": seq_length, "hidden_dim": hidden_dimensions, "epochs": num_epochs},
        best_iteration=num_epochs,
    )

    return suite


def sensitivity_analysis(
        features: pd.DataFrame,
        base_lsi: pd.Series,
        feature_cols: List[str],
        pct: float = 0.20,
        test_split_date: str | None = "2024-01-01",
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from .nsvm_model import NSVMAnomalyDetector
    from .nsvm_lsi import ECDFCalibration

    df = features[feature_cols].dropna(how="all").fillna(0.0)
    train_mask = df.index < pd.Timestamp(test_split_date) if test_split_date else pd.Series(True, index=df.index)
    if train_mask.sum() < 50:
        train_mask = pd.Series(True, index=df.index)

    train_idx = df.index[train_mask]
    X_train = df.loc[train_idx]

    base_params = {"lr": 0.001, "hidden_dim": 64, "seq_len": 14}
    rows = []
    columns: Dict[str, pd.Series] = {"lsi_base": base_lsi.reindex(features.index)}

    scenarios = [
        ("lr", +1, f"lr_+{int(pct * 100)}pct"), ("lr", -1, f"lr_-{int(pct * 100)}pct"),
        ("hidden_dim", +1, f"hidden_dim_+{int(pct * 100)}pct"), ("hidden_dim", -1, f"hidden_dim_-{int(pct * 100)}pct"),
        ("seq_len", +1, f"seq_len_+{int(pct * 100)}pct"), ("seq_len", -1, f"seq_len_-{int(pct * 100)}pct"),
    ]

    for param, sign, label in scenarios:
        perturbed = dict(base_params)
        new_val = perturbed[param] * (1.0 + sign * pct)
        if param in ("hidden_dim", "seq_len"):
            new_val = max(2, int(round(new_val)))
        perturbed[param] = new_val

        try:
            model = NSVMAnomalyDetector(
                seq_len=int(perturbed["seq_len"]),
                hidden_dim=int(perturbed["hidden_dim"]),
                epochs=15,
                lr=float(perturbed["lr"]),
                device="cpu",
            )
            model.fit(X_train)

            raw_full = pd.Series(model.predict(df), index=df.index)
            raw_train = raw_full.loc[raw_full.index.intersection(train_idx)].values
            cal = ECDFCalibration.fit(raw_train)
            lsi_arr = cal.transform(raw_full.values)
            lsi_series = pd.Series(lsi_arr, index=df.index, name="nsvm_lsi")

            lsi_series = apply_m4_penalty(lsi_series, features)
            lsi_series = exponential_smoothing(lsi_series, span=5)
            columns[label] = lsi_series
            aligned = pd.concat(
                [columns["lsi_base"].rename("base"), lsi_series.rename("scenario")],
                axis=1,
            ).dropna()
            diff = aligned["scenario"] - aligned["base"]
            corr = aligned["base"].corr(aligned["scenario"]) if len(aligned) > 1 else np.nan
            rows.append(
                {
                    "scenario": label,
                    "param": param,
                    "direction": sign,
                    "base_value": base_params[param],
                    "scenario_value": perturbed[param],
                    "mean_abs_diff": float(diff.abs().mean()),
                    "max_abs_diff": float(diff.abs().max()),
                    "q90_abs_diff": float(diff.abs().quantile(0.90)),
                    "corr_with_base": float(corr) if pd.notna(corr) else np.nan,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scenario": label,
                    "param": param,
                    "direction": sign,
                    "base_value": base_params[param],
                    "scenario_value": perturbed[param],
                    "mean_abs_diff": np.nan,
                    "max_abs_diff": np.nan,
                    "q90_abs_diff": np.nan,
                    "corr_with_base": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return pd.concat(columns, axis=1), pd.DataFrame(rows)


def detect_stress_episodes(lsi: pd.Series, threshold: float | None = None, min_episode_days: int = 5,
                           merge_gap_days: int = 5, top_k: int | None = None) -> pd.DataFrame:
    s = lsi.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "length_days", "max_lsi", "mean_lsi", "rank"])
    if threshold is None:
        threshold = float(np.nanpercentile(s.values, 90))

    above = s >= threshold
    if not above.any():
        return pd.DataFrame(columns=["start", "end", "length_days", "max_lsi", "mean_lsi", "rank"])

    dates = s.index[above]
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = cur_end = dates[0]
    for ts in dates[1:]:
        if (ts - cur_end).days <= merge_gap_days:
            cur_end = ts
        else:
            episodes.append((cur_start, cur_end))
            cur_start = cur_end = ts
    episodes.append((cur_start, cur_end))

    rows = [{"start": st, "end": en, "length_days": int((en - st).days) + 1, "max_lsi": float(s.loc[st:en].max()),
             "mean_lsi": float(s.loc[st:en].mean())} for st, en in episodes if
            int((en - st).days) + 1 >= min_episode_days]
    df = pd.DataFrame(rows).sort_values("max_lsi", ascending=False).reset_index(drop=True)
    if df.empty:
        return df.assign(rank=pd.Series(dtype=int))
    df["rank"] = np.arange(1, len(df) + 1)
    if top_k is not None:
        df = df.head(top_k)
    return df


def noise_breakdown(lsi_frame: pd.DataFrame) -> pd.DataFrame:
    contrib_cols = [f"contrib_{m}" for m in ("M1", "M2", "M3", "M4", "M5") if f"contrib_{m}" in lsi_frame.columns]
    if not contrib_cols:
        return pd.DataFrame(columns=["term", "component", "contribution", "share"])

    contribs = lsi_frame[contrib_cols].diff().dropna()
    rows = [{"term": col.replace("contrib_", ""), "component": "var", "contribution": float(contribs[col].var(ddof=0))}
            for col in contrib_cols]

    modules = [c.replace("contrib_", "") for c in contrib_cols]
    for i, mi in enumerate(modules):
        for mj in modules[i + 1:]:
            cov = float(contribs[f"contrib_{mi}"].cov(contribs[f"contrib_{mj}"], ddof=0))
            rows.append({"term": f"{mi}x{mj}", "component": "cov", "contribution": 2.0 * cov})

    df = pd.DataFrame(rows)
    total = df["contribution"].sum()
    df["share"] = df["contribution"] / total if total != 0.0 else 0.0
    return df.sort_values("contribution", ascending=False).reset_index(drop=True)


__all__ = ["MODULE_SCORE_FORMULA", "module_columns", "module_score", "all_module_scores", "apply_m4_penalty",
           "exponential_smoothing", "build_lsi_frame", "lsi_status", "aggregator_feature_columns", "fit_model_suite",
           "ModelResult", "MultiModelResult", "sensitivity_analysis", "detect_stress_episodes", "noise_breakdown"]
