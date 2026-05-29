"""GBM-based LSI: log-calibrated stress index and SHAP attribution.

NSVMRegressor is trained against
the proxy stress target and emits a real-valued prediction in the
target's scale.  This module converts those predictions into an LSI
on the same [0, 100] scale as the DSM-based LSI, using the same
log-linear anchors so the three indices are directly comparable on
the dashboard.

Per-module attribution uses SHAP values (GradientExplainer for NSVM
— sums over seq_len axis). Module
contribution is the magnitude of the SHAP value summed over the
module's features, normalised to a share that sums to 1 by row.

Workflow:

  pred = model.predict(X)
  LSI  = log_calibrate(pred, train_anchors)            ∈ [0, 100]

  shap_row = GradientExplainer(model).shap_values(x)   (per-feature, signed)
  share_M  = |shap_row[idx_M]| / Σ |shap_row|          ∈ [0, 1]
  contrib_M = share_M · LSI                            (Σ contrib = LSI)
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Log-calibration — identical to DSM's anomaly_score so the three LSIs are
# comparable on the same scale.
# ---------------------------------------------------------------------------


@dataclass
class LogCalibration:
    """Two-anchor log-linear stress calibrator.

    Fitted on the train-set distribution of raw predictions.  Maps any
    new raw value into ``[0, 100]`` monotonically without look-ahead.
    """

    q_low: float
    q_high: float
    q_low_quantile: float = 0.10
    q_high_quantile: float = 0.99
    fallback_rank: Optional[np.ndarray] = None  # train values for rank fallback

    @classmethod
    def fit(
        cls,
        train_raw: np.ndarray,
        q_low: float = 0.10,
        q_high: float = 0.99,
    ) -> "LogCalibration":
        """Pick the two anchors from the train distribution.

        Predictions for stress are positive; if the model occasionally
        outputs a negative number we shift the whole series before
        log-scaling so logs stay finite.  The shift is recorded inside
        ``q_low`` / ``q_high`` (they live in the *post-shift* space).
        """

        arr = np.asarray(train_raw, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            return cls(q_low=0.0, q_high=1.0, fallback_rank=arr)

        # Shift to strictly positive: log requires it.  We keep the
        # offset as part of the anchors and reapply it at transform
        # time.  Subtracting min - eps guarantees min(shifted) > 0.
        offset = float(arr.min()) - 1e-6 if arr.min() <= 0 else 0.0
        shifted = arr - offset

        ql = float(np.quantile(shifted, q_low))
        qh = float(np.quantile(shifted, q_high))
        if qh <= ql * 1.000001:
            # Collapsed predictor — fall back to rank-CDF on raw.
            return cls(
                q_low=float("nan"),
                q_high=float("nan"),
                q_low_quantile=q_low,
                q_high_quantile=q_high,
                fallback_rank=arr,
            )

        # Bake the offset into q_low by storing the post-shift anchors
        # but ALSO storing the offset itself.  Trick: encode offset as a
        # negative sentinel through the fallback_rank slot.  Cleaner:
        # add it as an explicit field on the dataclass.
        cal = cls(
            q_low=ql,
            q_high=qh,
            q_low_quantile=q_low,
            q_high_quantile=q_high,
        )
        cal._offset = offset  # type: ignore[attr-defined]
        return cal

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Apply the log-linear map to a vector of raw predictions."""

        arr = np.asarray(raw, dtype=float)

        # Fallback: empirical CDF on stored train values.
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


# ---------------------------------------------------------------------------
# SHAP attribution for tree models
# ---------------------------------------------------------------------------


def shap_module_attribution(
        model: Any,
        X: pd.DataFrame,
        feature_to_module: Dict[str, str],
        lsi: pd.Series,
) -> pd.DataFrame:
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
            print("Запуск SHAP GradientExplainer для NSVM...")

            X_tensor = model._create_sequences(X.values).to(model.device)
            bg_idx = np.random.choice(len(X_tensor), size=min(100, len(X_tensor)), replace=False)
            background_tensor = X_tensor[bg_idx]

            # Простейший враппер: отрезаем log_var, отдаем только mu для вычисления градиентов
            class ModelWrapper(torch.nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net

                def forward(self, x):
                    mu, _ = self.net(x)
                    return mu.unsqueeze(1)

            explainer_net = ModelWrapper(model.model).to(model.device)
            explainer_net.eval()

            # GradientExplainer требует, чтобы requires_grad было True для входа внутри
            explainer = _shap.GradientExplainer(explainer_net, background_tensor)
            shap_values = explainer.shap_values(X_tensor)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # shap_values: (batch, seq_len, features) -> сворачиваем по оси времени
            sv = np.array(shap_values)
            # Убираем лишние размерности (batch, seq_len, features, 1) -> (batch, seq_len, features)
            if sv.ndim == 4:
                sv = sv.squeeze(-1)
            # (batch, seq_len, features) -> суммируем по seq_len (axis=1) -> (batch, features)
            if sv.ndim == 3:
                shap_arr = np.abs(sv).sum(axis=1)
            elif sv.ndim == 2:
                shap_arr = np.abs(sv)
            else:
                shap_arr = np.abs(sv)
            print("SHAP GradientExplainer успешно отработал!")

        else:
            # Fallback for non-NSVM tree models (unused in current pipeline)
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


# ---------------------------------------------------------------------------
# Top-level convenience: fit calibrator, build LSI, decompose by module
# ---------------------------------------------------------------------------


def gbm_lsi(
    model: Any,
    X_full: pd.DataFrame,
    train_idx: pd.Index,
    feature_to_module: Dict[str, str],
    q_low: float = 0.10,
    q_high: float = 0.99,
) -> Tuple[pd.Series, pd.DataFrame, LogCalibration]:
    """Build a GBM-based LSI on the full timeline.

    Steps:
      1. Predict on every row of ``X_full``.
      2. Fit a :class:`LogCalibration` on predictions restricted to
         ``train_idx``.
      3. Apply the calibrator to all predictions → LSI ∈ [0, 100].
      4. Decompose the LSI per module via SHAP.

    Returns
    -------
    pd.Series
        LSI on the full index.
    pd.DataFrame
        Module attribution frame (same schema as DSM).
    LogCalibration
        The fitted calibrator (for diagnostics / sensitivity).
    """

    raw_full = pd.Series(model.predict(X_full), index=X_full.index)
    raw_train = raw_full.loc[raw_full.index.intersection(train_idx)].values
    cal = LogCalibration.fit(raw_train, q_low=q_low, q_high=q_high)
    lsi_arr = cal.transform(raw_full.values)
    lsi = pd.Series(lsi_arr, index=X_full.index, name="gbm_lsi")
    attr = shap_module_attribution(model, X_full, feature_to_module, lsi)
    return lsi, attr, cal


__all__ = [
    "LogCalibration",
    "gbm_lsi",
    "shap_module_attribution",
]
