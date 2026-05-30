from __future__ import annotations

from dataclasses import dataclass
import torch
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _shap = None
    _SHAP_AVAILABLE = False


@dataclass
class LogCalibration:
    """Two-anchor log-linear stress calibrator."""
    q_low: float
    q_high: float
    q_low_quantile: float = 0.10
    q_high_quantile: float = 0.99
    fallback_rank: Optional[np.ndarray] = None

    @classmethod
    def fit(cls, train_raw: np.ndarray, q_low: float = 0.10, q_high: float = 0.99) -> "LogCalibration":
        arr = np.asarray(train_raw, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            return cls(q_low=0.0, q_high=1.0, fallback_rank=arr)

        offset = float(arr.min()) - 1e-6 if arr.min() <= 0 else 0.0
        shifted = arr - offset

        ql = float(np.quantile(shifted, q_low))
        qh = float(np.quantile(shifted, q_high))
        if qh <= ql * 1.000001:
            return cls(q_low=float("nan"), q_high=float("nan"), q_low_quantile=q_low, q_high_quantile=q_high, fallback_rank=arr)

        cal = cls(q_low=ql, q_high=qh, q_low_quantile=q_low, q_high_quantile=q_high)
        cal._offset = offset
        return cal

    def transform(self, raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw, dtype=float)

        if not np.isfinite(self.q_low) or not np.isfinite(self.q_high):
            base = self.fallback_rank
            if base is None or base.size == 0:
                return np.zeros_like(arr)
            sorted_base = np.sort(base)
            rank_lo = np.searchsorted(sorted_base, arr, side="left")
            rank_hi = np.searchsorted(sorted_base, arr, side="right")
            ranks = (rank_lo + rank_hi) / 2.0 / sorted_base.size
            return np.clip(ranks * 100.0, 0.0, 100.0)

        offset = getattr(self, "_offset", 0.0)
        shifted = arr - offset
        shifted = np.clip(shifted, self.q_low * 1e-3, None)
        log_low = np.log(self.q_low)
        log_high = np.log(self.q_high)
        scores = 100.0 * (np.log(shifted) - log_low) / (log_high - log_low)
        return np.clip(scores, 0.0, 100.0)


def shap_module_attribution(model: Any, X: pd.DataFrame, feature_to_module: Dict[str, str], lsi: pd.Series) -> pd.DataFrame:
    canonical_modules = ("M1", "M2", "M3", "M4", "M5")
    out = pd.DataFrame(index=X.index)
    out["dsm_anomaly"] = lsi.reindex(X.index).fillna(0.0)
    for module in canonical_modules:
        out[f"share_{module}"] = 0.0
        out[f"contrib_{module}"] = 0.0

    if not _SHAP_AVAILABLE:
        return out

    try:
        if hasattr(model, '_create_sequences'):
            print("Запуск SHAP GradientExplainer для NSVM (NLL)...")

            X_tensor = model._create_sequences(X.values).to(model.device)
            bg_idx = np.random.choice(len(X_tensor), size=min(100, len(X_tensor)), replace=False)
            background_tensor = X_tensor[bg_idx]

            class ModelWrapper(torch.nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net

                def forward(self, x):
                    mu, log_var = self.net(x)
                    var = torch.exp(log_var)
                    target = x[:, -1, :]
                    # Возвращаем NLL для вычисления атрибуции аномалии
                    nll = 0.5 * torch.mean(log_var + ((target - mu) ** 2) / (var + 1e-6), dim=1)
                    return nll.unsqueeze(1)

            explainer_net = ModelWrapper(model.model).to(model.device)
            explainer_net.eval()

            explainer = _shap.GradientExplainer(explainer_net, background_tensor)
            shap_values = explainer.shap_values(X_tensor)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            sv = np.array(shap_values)
            if sv.ndim == 4:
                sv = sv.squeeze(-1)
            if sv.ndim == 3:
                shap_arr = np.abs(sv).sum(axis=1)
            elif sv.ndim == 2:
                shap_arr = np.abs(sv)
            else:
                shap_arr = np.abs(sv)
            print("SHAP GradientExplainer успешно отработал!")

        else:
            explainer = _shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_arr = np.abs(np.asarray(shap_values, dtype=float))

    except Exception as e:
        print(f"\n!!! ОШИБКА SHAP !!! : {e}\n")
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
        m = feature_to_module.get(col, "OTHER")
        if m in module_idx:
            module_idx[m].append(i)

    total_abs = shap_arr.sum(axis=1)
    total_abs_safe = np.where(total_abs < 1e-12, 1.0, total_abs)

    for module in canonical_modules:
        idx = module_idx[module]
        if not idx:
            continue
        sumsq = shap_arr[:, idx].sum(axis=1)
        share = sumsq / total_abs_safe
        out[f"share_{module}"] = share
        out[f"contrib_{module}"] = share * out["dsm_anomaly"].values

    return out


def gbm_lsi(model: Any, X_full: pd.DataFrame, train_idx: pd.Index, feature_to_module: Dict[str, str], q_low: float = 0.10, q_high: float = 0.99) -> Tuple[pd.Series, pd.DataFrame, LogCalibration]:
    raw_full = pd.Series(model.predict(X_full), index=X_full.index)
    raw_train = raw_full.loc[raw_full.index.intersection(train_idx)].values
    cal = LogCalibration.fit(raw_train, q_low=q_low, q_high=q_high)
    lsi_arr = cal.transform(raw_full.values)
    lsi = pd.Series(lsi_arr, index=X_full.index, name="gbm_lsi")
    attr = shap_module_attribution(model, X_full, feature_to_module, lsi)
    return lsi, attr, cal


__all__ = ["LogCalibration", "gbm_lsi", "shap_module_attribution"]