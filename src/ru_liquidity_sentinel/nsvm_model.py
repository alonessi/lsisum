# nsvm_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin


class NSVMArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, seq_len=10):
        super().__init__()
        self.seq_len = seq_len

        # Детерминированный путь (mean)
        self.lstm_mean = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Путь волатильности (variance)
        self.lstm_vol = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True)
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Выдает log-variance
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out_mean, _ = self.lstm_mean(x)
        out_vol, _ = self.lstm_vol(x)

        # Берем только последний шаг (many-to-one)
        last_step_mean = out_mean[:, -1, :]
        last_step_vol = out_vol[:, -1, :]

        mu = self.mean_head(last_step_mean).squeeze(-1)
        log_var = self.vol_head(last_step_vol).squeeze(-1)

        return mu, log_var


class NSVMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-Learn обертка для NSVM, формирующая окна (seq_len) под капотом."""

    def __init__(self, seq_len=10, hidden_dim=64, epochs=50, lr=1e-3, batch_size=32, device="cpu"):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _create_sequences(self, X, y=None):
        X_seq, y_seq = [], []
        # Добавляем паддинг для начальных значений, чтобы длина выхода совпадала с длиной входа
        pad_size = self.seq_len - 1
        X_padded = np.vstack([np.zeros((pad_size, X.shape[1])), X])

        for i in range(len(X)):
            X_seq.append(X_padded[i: i + self.seq_len])
            if y is not None:
                y_seq.append(y[i])

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        if y is not None:
            y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)
            return X_tensor, y_tensor
        return X_tensor

    def fit(self, X, y, eval_set=None, **kwargs):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values

        X_tensor, y_tensor = self._create_sequences(X, y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = NSVMArchitecture(input_dim=X.shape[1], hidden_dim=self.hidden_dim, seq_len=self.seq_len)
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Loss: Negative Log-Likelihood для Гауссианы
        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()

                mu, log_var = self.model(batch_X)
                var = torch.exp(log_var)

                # NLL Loss: 0.5 * [ log(var) + (y - mu)^2 / var ]
                loss = 0.5 * torch.mean(log_var + ((batch_y - mu) ** 2) / (var + 1e-6))

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        self.model.eval()
        X_tensor = self._create_sequences(X)

        # Для прогноза LSI используем детерминированную компоненту (mu)
        with torch.no_grad():
            mu, _ = self.model(X_tensor.to(self.device))

        return mu.cpu().numpy()

    def partial_fit(self, X, y, epochs=10, lr=1e-4):
        """
        Дообучение модели на новых данных (адаптация к разладке).
        """
        if self.model is None:
            raise RuntimeError("Модель должна быть сначала обучена через .fit()")

        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values

        # Создаем тензоры
        X_tensor, y_tensor = self._create_sequences(X, y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 1. ЗАМОРОЗКА БАЗОВЫХ СЛОЕВ (LSTM)
        # Мы хотим, чтобы память о долгосрочных паттернах осталась,
        # но выходные слои подстроились под новые уровни (bias/scale)
        for param in self.model.lstm_mean.parameters():
            param.requires_grad = False
        for param in self.model.lstm_vol.parameters():
            param.requires_grad = False

        # 2. НАСТРОЙКА ОПТИМИЗАТОРА С МАЛЕНЬКИМ LR
        # Обучаем только mean_head и vol_head
        trainable_params = (
                list(self.model.mean_head.parameters()) +
                list(self.model.vol_head.parameters())
        )
        optimizer = optim.Adam(trainable_params, lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()

                mu, log_var = self.model(batch_X)
                var = torch.exp(log_var)

                # Используем тот же NLL Loss
                loss = 0.5 * torch.mean(log_var + ((batch_y - mu) ** 2) / (var + 1e-6))

                loss.backward()
                optimizer.step()

        # 3. РАЗМОРОЗКА (чтобы не сломать будущие вызовы fit, если они будут)
        for param in self.model.parameters():
            param.requires_grad = True

        return self