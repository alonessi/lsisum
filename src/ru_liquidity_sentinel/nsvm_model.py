import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator


class NSVMArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, seq_len=14):
        super().__init__()
        self.seq_len = seq_len

        # Общий энкодер для формирования эмбеддинга рыночного состояния
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Декодеры теперь выдают векторы размерности input_dim
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        last_step_hidden = out[:, -1, :]  # Берем скрытое состояние последнего шага

        mu = self.mean_head(last_step_hidden)
        log_var = self.vol_head(last_step_hidden)
        return mu, log_var


class NSVMAnomalyDetector(BaseEstimator):
    """Unsupervised NSVM: оценивает вероятность (Negative Log-Likelihood) текущего состояния."""

    def __init__(self, seq_len=14, hidden_dim=64, epochs=30, lr=1e-3, batch_size=32, device="cpu"):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _create_sequences(self, X):
        X_seq = []
        pad_size = self.seq_len - 1
        X_padded = np.vstack([np.zeros((pad_size, X.shape[1])), X])

        for i in range(len(X)):
            X_seq.append(X_padded[i: i + self.seq_len])

        return torch.tensor(np.array(X_seq), dtype=torch.float32)

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame): X = X.values

        X_tensor = self._create_sequences(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = NSVMArchitecture(input_dim=X.shape[1], hidden_dim=self.hidden_dim, seq_len=self.seq_len)
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                optimizer.zero_grad()

                mu, log_var = self.model(batch_X)
                var = torch.exp(log_var)

                # Таргет - это сами фичи на последнем шаге окна
                target = batch_X[:, -1, :]

                # Многомерный NLL Loss
                loss = 0.5 * torch.mean(log_var + ((target - mu) ** 2) / (var + 1e-6))

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """Возвращает NLL (Anomaly Score) для каждого наблюдения."""
        if isinstance(X, pd.DataFrame): X = X.values
        self.model.eval()
        X_tensor = self._create_sequences(X).to(self.device)

        with torch.no_grad():
            mu, log_var = self.model(X_tensor)
            var = torch.exp(log_var)
            target = X_tensor[:, -1, :]

            # Считаем NLL по каждому сэмплу (усредняем по фичам)
            nll = 0.5 * torch.mean(log_var + ((target - mu) ** 2) / (var + 1e-6), dim=1)

        return nll.cpu().numpy()

    def partial_fit(self, X, epochs=10, lr=1e-4):
        if self.model is None:
            raise RuntimeError("Модель должна быть обучена через .fit()")

        if isinstance(X, pd.DataFrame): X = X.values
        X_tensor = self._create_sequences(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Морозим память (LSTM), обучаем только проекции
        for param in self.model.lstm.parameters():
            param.requires_grad = False

        trainable_params = list(self.model.mean_head.parameters()) + list(self.model.vol_head.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                optimizer.zero_grad()

                mu, log_var = self.model(batch_X)
                var = torch.exp(log_var)
                target = batch_X[:, -1, :]

                loss = 0.5 * torch.mean(log_var + ((target - mu) ** 2) / (var + 1e-6))
                loss.backward()
                optimizer.step()

        for param in self.model.parameters():
            param.requires_grad = True

        return self