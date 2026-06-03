from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset


class NSVMArchitecture(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, seq_len: int = 14):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, input_dim),
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last_step_hidden = self.dropout(out[:, -1, :])
        mu = self.mean_head(last_step_hidden)
        log_var = self.vol_head(last_step_hidden)
        return mu, log_var


class NSVMAnomalyDetector(BaseEstimator):
    """Unsupervised LSTM forecaster used as an NSVM-style anomaly detector."""

    def __init__(
        self,
        seq_len: int = 14,
        hidden_dim: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model: NSVMArchitecture | None = None

    @staticmethod
    def _as_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    def _create_sequences(
        self,
        X: pd.DataFrame | np.ndarray,
        context: pd.DataFrame | np.ndarray | None = None,
    ) -> torch.Tensor:
        X_arr = self._as_array(X).astype(float, copy=False)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or DataFrame")
        if len(X_arr) == 0:
            return torch.empty((0, self.seq_len, X_arr.shape[1]), dtype=torch.float32)

        if context is None:
            left_context = np.zeros((self.seq_len, X_arr.shape[1]), dtype=float)
        else:
            ctx = self._as_array(context).astype(float, copy=False)
            if ctx.ndim != 2 or ctx.shape[1] != X_arr.shape[1]:
                raise ValueError("context must be 2D and match X feature count")
            left_context = ctx[-self.seq_len :]
            if len(left_context) < self.seq_len:
                pad = np.zeros((self.seq_len - len(left_context), X_arr.shape[1]), dtype=float)
                left_context = np.vstack([pad, left_context])

        padded = np.vstack([left_context, X_arr])
        sequences = [padded[i : i + self.seq_len] for i in range(len(X_arr))]
        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None, **kwargs: Any):
        X_arr = self._as_array(X).astype(float, copy=False)
        X_tensor = self._create_sequences(X_arr)
        target_tensor = torch.tensor(X_arr, dtype=torch.float32)

        if len(X_tensor) > self.seq_len:
            X_tensor = X_tensor[self.seq_len :]
            target_tensor = target_tensor[self.seq_len :]

        dataset = TensorDataset(X_tensor, target_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = NSVMArchitecture(
            input_dim=X_arr.shape[1],
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
        )
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        for _epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_target in loader:
                batch_X = batch_X.to(self.device)
                batch_target = batch_target.to(self.device)
                optimizer.zero_grad()
                mu, log_var = self.model(batch_X)
                log_var = torch.clamp(log_var, min=-4.0, max=4.0)
                var = torch.exp(log_var)
                loss = 0.5 * torch.mean(log_var + ((batch_target - mu) ** 2) / var)
                loss.backward()
                optimizer.step()

        return self

    def _predict_tensors(
        self,
        X: pd.DataFrame | np.ndarray,
        context: pd.DataFrame | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        X_arr = self._as_array(X).astype(float, copy=False)
        self.model.eval()
        X_tensor = self._create_sequences(X_arr, context=context).to(self.device)
        target_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mu, log_var = self.model(X_tensor)
        return target_tensor, mu, log_var

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        context: pd.DataFrame | np.ndarray | None = None,
    ) -> np.ndarray:
        target_tensor, mu, log_var = self._predict_tensors(X, context=context)
        log_var = torch.clamp(log_var, min=-4.0, max=4.0)
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + ((target_tensor - mu) ** 2) / var)
        return torch.mean(nll, dim=1).cpu().numpy()

    def get_nll_components(
        self,
        X: pd.DataFrame | np.ndarray,
        context: pd.DataFrame | np.ndarray | None = None,
    ) -> np.ndarray:
        target_tensor, mu, log_var = self._predict_tensors(X, context=context)
        log_var = torch.clamp(log_var, min=-4.0, max=4.0)
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + ((target_tensor - mu) ** 2) / var)
        return nll.cpu().numpy()

    def partial_fit(
        self,
        X: pd.DataFrame | np.ndarray,
        epochs: int = 10,
        lr: float = 1e-4,
        context: pd.DataFrame | np.ndarray | None = None,
    ):
        if self.model is None:
            raise RuntimeError("Model must be fitted before partial_fit")

        X_arr = self._as_array(X).astype(float, copy=False)
        X_tensor = self._create_sequences(X_arr, context=context)
        target_tensor = torch.tensor(X_arr, dtype=torch.float32)

        if context is None and len(X_tensor) > self.seq_len:
            X_tensor = X_tensor[self.seq_len :]
            target_tensor = target_tensor[self.seq_len :]

        dataset = TensorDataset(X_tensor, target_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for param in self.model.lstm.parameters():
            param.requires_grad = False

        trainable_params = list(self.model.mean_head.parameters()) + list(self.model.vol_head.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr)

        self.model.train()
        for _epoch in range(epochs):
            for batch_X, batch_target in loader:
                batch_X = batch_X.to(self.device)
                batch_target = batch_target.to(self.device)
                optimizer.zero_grad()
                mu, log_var = self.model(batch_X)
                log_var = torch.clamp(log_var, min=-4.0, max=4.0)
                var = torch.exp(log_var)
                loss = 0.5 * torch.mean(log_var + ((batch_target - mu) ** 2) / var)
                loss.backward()
                optimizer.step()

        for param in self.model.parameters():
            param.requires_grad = True

        return self

    def save_weights(self, path: str | Path, feature_cols: list[str] | None = None, extra: dict | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "input_dim": self.model.mean_head[-1].out_features,
            "feature_cols": list(feature_cols or []),
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)

    @classmethod
    def load_weights(cls, path: str | Path, device: str = "cpu") -> "NSVMAnomalyDetector":
        payload = torch.load(path, map_location=device)
        detector = cls(
            seq_len=int(payload["seq_len"]),
            hidden_dim=int(payload["hidden_dim"]),
            epochs=int(payload.get("epochs", 30)),
            lr=float(payload.get("lr", 1e-3)),
            batch_size=int(payload.get("batch_size", 32)),
            device=device,
        )
        detector.model = NSVMArchitecture(
            input_dim=int(payload["input_dim"]),
            hidden_dim=detector.hidden_dim,
            seq_len=detector.seq_len,
        )
        detector.model.load_state_dict(payload["state_dict"])
        detector.model.to(device)
        detector.feature_cols_ = list(payload.get("feature_cols") or [])
        detector.extra_ = dict(payload.get("extra") or {})
        return detector
