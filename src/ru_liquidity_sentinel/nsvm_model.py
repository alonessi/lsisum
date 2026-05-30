import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator


class NSVMArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, seq_len=14):  # Уменьшили размерность
        super().__init__()
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)  # Жесткий дропаут для борьбы с зубрежкой

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, input_dim)
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_hidden = out[:, -1, :]
        last_step_hidden = self.dropout(last_step_hidden)  # Прореживаем скрытое состояние

        mu = self.mean_head(last_step_hidden)
        log_var = self.vol_head(last_step_hidden)
        return mu, log_var


class NSVMAnomalyDetector(BaseEstimator):
    """Unsupervised NSVM: Forecasting anomaly detector."""

    def __init__(self, seq_len=14, hidden_dim=64, epochs=30, lr=1e-3, batch_size=32, device="cpu"):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _create_sequences(self, X):
        # Строгое прогнозирование: чтобы предсказать X[i], берем историю ДО него
        pad_size = self.seq_len
        X_padded = np.vstack([np.zeros((pad_size, X.shape[1])), X])

        X_seq = []
        for i in range(len(X)):
            # Берем seq_len шагов в прошлом. X_padded[i+pad_size] - это текущий X[i]
            # Поэтому берем срез от i до i + seq_len
            X_seq.append(X_padded[i: i + self.seq_len])

        return torch.tensor(np.array(X_seq), dtype=torch.float32)

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame): X = X.values

        X_tensor = self._create_sequences(X)
        target_tensor = torch.tensor(X, dtype=torch.float32)  # Истинный X_t

        dataset = TensorDataset(X_tensor, target_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = NSVMArchitecture(input_dim=X.shape[1], hidden_dim=self.hidden_dim, seq_len=self.seq_len)
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_target in loader:
                batch_X, batch_target = batch_X.to(self.device), batch_target.to(self.device)
                optimizer.zero_grad()

                mu, log_var = self.model(batch_X)

                # ЗАЩИТА ОТ КОЛЛАПСА ДИСПЕРСИИ И NaN
                log_var = torch.clamp(log_var, min=-4.0, max=4.0)
                var = torch.exp(log_var)

                # NLL Loss
                loss = 0.5 * torch.mean(log_var + ((batch_target - mu) ** 2) / var)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        self.model.eval()
        X_tensor = self._create_sequences(X).to(self.device)
        target_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mu, _ = self.model(X_tensor)
            # ИСПРАВЛЕНО: Вместо NLL возвращаем MAE (абсолютную ошибку).
            # Она строго > 0 и не взрывается от квадратов и деления на дисперсию.
            mae = torch.mean(torch.abs(target_tensor - mu), dim=1)

        return mae.cpu().numpy()

    def get_nll_components(self, X):
        """Возвращает аналитический MAE для каждой фичи (используется для атрибуции)."""
        if isinstance(X, pd.DataFrame): X = X.values
        self.model.eval()
        X_tensor = self._create_sequences(X).to(self.device)
        target_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mu, _ = self.model(X_tensor)
            # ИСПРАВЛЕНО: Возвращаем матрицу абсолютных ошибок для SHAP-атрибуции
            mae_components = torch.abs(target_tensor - mu)

        return mae_components.cpu().numpy()

    def partial_fit(self, X, epochs=10, lr=1e-4):
        if self.model is None:
            raise RuntimeError("Модель должна быть обучена через .fit()")

        if isinstance(X, pd.DataFrame): X = X.values
        X_tensor = self._create_sequences(X)
        target_tensor = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, target_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for param in self.model.lstm.parameters():
            param.requires_grad = False

        trainable_params = list(self.model.mean_head.parameters()) + list(self.model.vol_head.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_target in loader:
                batch_X, batch_target = batch_X.to(self.device), batch_target.to(self.device)
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