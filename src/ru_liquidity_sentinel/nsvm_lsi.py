from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import shap as _shap

    _SHAP_AVAILABLE = True
except ImportError:
    _shap = None
    _SHAP_AVAILABLE = False


@dataclass
class ECDFCalibration:
    """Empirical CDF calibrator with calm and stressed quantile anchors."""

    train_scores: np.ndarray
    q_low: float = 0.20
    q_high: float = 0.995

    @classmethod
    def fit(
        cls,
        train_raw: np.ndarray,
        q_low: float = 0.20,
        q_high: float = 0.995,
        **kwargs,
    ) -> "ECDFCalibration":
        if not 0.0 <= q_low < q_high <= 1.0:
            raise ValueError("Expected quantiles to satisfy 0 <= q_low < q_high <= 1")
        arr = np.asarray(train_raw, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0])
        return cls(train_scores=np.sort(arr), q_low=q_low, q_high=q_high)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw, dtype=float)
        if self.train_scores.size == 0:
            return np.zeros_like(arr)

        rank_lo = np.searchsorted(self.train_scores, arr, side="left")
        rank_hi = np.searchsorted(self.train_scores, arr, side="right")
        percentile = (rank_lo + rank_hi) / 2.0 / self.train_scores.size

        q_low = float(getattr(self, "q_low", 0.20))
        q_high = float(getattr(self, "q_high", 0.995))
        if not 0.0 <= q_low < q_high <= 1.0:
            q_low, q_high = 0.20, 0.995

        scaled = (percentile - q_low) / (q_high - q_low)
        return np.clip(scaled * 100.0, 0.0, 100.0)


def estimate_module_attribution(
    model: Any,
    X: pd.DataFrame,
    feature_to_module: Dict[str, str],
    lsi: pd.Series,
) -> pd.DataFrame:
    """Estimate module shares from analytical NLL components or SHAP values."""

    canonical_modules = ("M1", "M2", "M3", "M4", "M5")
    out = pd.DataFrame(index=X.index)
    out["dsm_anomaly"] = lsi.reindex(X.index).fillna(0.0)

    for module in canonical_modules:
        out[f"share_{module}"] = 0.0
        out[f"contrib_{module}"] = 0.0

    try:
        if hasattr(model, "get_nll_components"):
            attribution_arr = np.abs(model.get_nll_components(X))
        else:
            explainer = _shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            attribution_arr = np.abs(np.asarray(shap_values, dtype=float))
    except Exception:
        present = {feature_to_module.get(c) for c in X.columns}
        present = {m for m in present if m in canonical_modules}
        if present:
            equal = 1.0 / len(present)
            for module in present:
                out[f"share_{module}"] = equal
                out[f"contrib_{module}"] = equal * out["dsm_anomaly"].values
        return out

    feature_cols = list(X.columns)
    module_idx = {m: [] for m in canonical_modules}

    for i, col in enumerate(feature_cols):
        module = feature_to_module.get(col, "OTHER")
        if module in module_idx:
            module_idx[module].append(i)

    total_abs = attribution_arr.sum(axis=1)
    total_abs_safe = np.where(total_abs < 1e-12, 1.0, total_abs)

    for module in canonical_modules:
        idx = module_idx[module]
        if not idx:
            continue
        module_abs = attribution_arr[:, idx].sum(axis=1)
        share = module_abs / total_abs_safe
        out[f"share_{module}"] = share
        out[f"contrib_{module}"] = share * out["dsm_anomaly"].values

    return out


shap_module_attribution = estimate_module_attribution


def nsvm_lsi(
    model: Any,
    X_full: pd.DataFrame,
    train_idx: pd.Index,
    feature_to_module: Dict[str, str],
    q_low: float = 0.20,
    q_high: float = 0.995,
) -> Tuple[pd.Series, pd.DataFrame, ECDFCalibration]:
    """Build an ECDF-calibrated LSI series and module attribution."""

    raw_full = pd.Series(model.predict(X_full), index=X_full.index)

    seq_len = getattr(model, "seq_len", 14)
    if len(raw_full) > seq_len:
        raw_full.iloc[:seq_len] = np.nan

    raw_full = raw_full.ewm(span=7, adjust=False).mean()

    raw_train = raw_full.loc[raw_full.index.intersection(train_idx)].dropna().values
    cal = ECDFCalibration.fit(raw_train, q_low=q_low, q_high=q_high)

    lsi_arr = cal.transform(raw_full.bfill().ffill().values)
    lsi = pd.Series(lsi_arr, index=X_full.index, name="nsvm_lsi")

    attr = estimate_module_attribution(model, X_full, feature_to_module, lsi)
    return lsi, attr, cal


__all__ = [
    "ECDFCalibration",
    "nsvm_lsi",
    "estimate_module_attribution",
    "shap_module_attribution",
]
