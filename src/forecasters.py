"""
ARA-PPO v3 — Base forecasters for the hybrid ensemble
======================================================
Each forecaster predicts (μ, σ) for the next-day log-return.

Forecasters
-----------
  LSTMForecaster          temporal momentum, autoregressive returns sequences
  TransformerForecaster   long-range dependencies via self-attention
  XGBoostForecaster       tabular / non-linear feature interactions
  CNNForecaster           local pattern extraction in time series
  GARCHForecaster         conditional volatility (σ only; μ from sample mean)

All forecasters share a common interface so they're easily swappable in the
RegimeAwareEnsemble.  Train per walk-forward split (no lookahead).

Common API
----------
  fit(df_train)                        # returns self
  predict_mu_sigma(df) -> (T, 2)       # column 0 = μ, column 1 = σ
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseForecaster(ABC):
    """All forecasters predict next-day log-return (μ) and its std (σ)."""

    name: str = "base"

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> "BaseForecaster": ...

    @abstractmethod
    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return (T, 2) array.  Column 0: predicted μ.  Column 1: predicted σ.
        T must equal len(df).  First lookback rows may be filled with sample
        mean / std (no actual prediction available).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build (X, y) sequences from a feature DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    lookback: int,
    target_col: str = "log_return",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X, y) pairs for next-day return prediction.

      X[i] = features over [i, i+lookback)         shape: (lookback, n_features)
      y[i] = target_col at i+lookback              shape: scalar

    Returns arrays X (N, lookback, n_features) and y (N,).
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].fillna(0).values.astype(np.float32)
    N = len(df) - lookback
    if N <= 0:
        return np.empty((0, lookback, len(feature_cols)), dtype=np.float32), \
               np.empty((0,), dtype=np.float32)
    Xs = np.stack([X[i:i+lookback] for i in range(N)], axis=0)
    ys = y[lookback:lookback+N]
    return Xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# 1. LSTM Forecaster — temporal momentum
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(
        self,
        feature_cols: list,
        lookback: int = 60,
        hidden: int   = 64,
        n_layers: int = 2,
        epochs: int   = 15,
        batch_size: int = 64,
        lr: float     = 1e-3,
        device: str   = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.lookback     = lookback
        self.hidden       = hidden
        self.n_layers     = n_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else device if device != "auto" else "cpu")
        self.net: Optional[_LSTMNet]            = None
        self.feat_mean: Optional[np.ndarray]    = None
        self.feat_std:  Optional[np.ndarray]    = None
        self.sigma:     float                   = 0.02

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / (self.feat_std + 1e-8)

    def fit(self, df_train: pd.DataFrame) -> "LSTMForecaster":
        # Z-score features on training only
        feats = df_train[self.feature_cols].values.astype(np.float32)
        self.feat_mean = feats.mean(0, keepdims=True)
        self.feat_std  = feats.std(0, keepdims=True) + 1e-8

        X, y = _build_sequences(df_train, self.feature_cols, self.lookback)
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self
        X = (X - self.feat_mean) / self.feat_std
        Xt = th.from_numpy(X).to(self.device)
        yt = th.from_numpy(y).to(self.device)

        self.net = _LSTMNet(len(self.feature_cols), self.hidden, self.n_layers).to(self.device)
        opt = th.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.net(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

        # Estimate residual std on training data
        self.net.eval()
        with th.no_grad():
            preds = self.net(Xt).cpu().numpy()
        self.sigma = float(np.std(y - preds) + 1e-6)
        return self

    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        out = np.zeros((T, 2), dtype=np.float32)
        out[:, 1] = self.sigma  # constant σ from training residuals

        if self.net is None or T <= self.lookback:
            out[:, 0] = 0.0
            return out

        feats = df[self.feature_cols].values.astype(np.float32)
        feats = (feats - self.feat_mean) / self.feat_std
        # Sliding windows
        N = T - self.lookback
        X = np.stack([feats[i:i+self.lookback] for i in range(N)], axis=0)
        Xt = th.from_numpy(X).to(self.device)
        self.net.eval()
        with th.no_grad():
            preds = self.net(Xt).cpu().numpy()
        out[self.lookback:self.lookback+N, 0] = preds
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transformer Forecaster — long-range dependencies
# ─────────────────────────────────────────────────────────────────────────────

class _TransformerNet(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, n_layers=2, lookback=60):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(th.randn(1, lookback, d_model) * 0.02)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)


class TransformerForecaster(BaseForecaster):
    name = "transformer"

    def __init__(
        self,
        feature_cols: list,
        lookback: int = 60,
        d_model: int  = 32,
        nhead: int    = 4,
        n_layers: int = 2,
        epochs: int   = 15,
        batch_size: int = 64,
        lr: float     = 5e-4,
        device: str   = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.lookback     = lookback
        self.d_model      = d_model
        self.nhead        = nhead
        self.n_layers     = n_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else device if device != "auto" else "cpu")
        self.net: Optional[_TransformerNet] = None
        self.feat_mean = None
        self.feat_std  = None
        self.sigma     = 0.02

    def fit(self, df_train: pd.DataFrame) -> "TransformerForecaster":
        feats = df_train[self.feature_cols].values.astype(np.float32)
        self.feat_mean = feats.mean(0, keepdims=True)
        self.feat_std  = feats.std(0, keepdims=True) + 1e-8

        X, y = _build_sequences(df_train, self.feature_cols, self.lookback)
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self
        X = (X - self.feat_mean) / self.feat_std
        Xt = th.from_numpy(X).to(self.device)
        yt = th.from_numpy(y).to(self.device)

        self.net = _TransformerNet(
            len(self.feature_cols), self.d_model, self.nhead, self.n_layers, self.lookback,
        ).to(self.device)
        opt = th.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.mse_loss(self.net(xb), yb)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

        self.net.eval()
        with th.no_grad():
            preds = self.net(Xt).cpu().numpy()
        self.sigma = float(np.std(y - preds) + 1e-6)
        return self

    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        out = np.zeros((T, 2), dtype=np.float32)
        out[:, 1] = self.sigma
        if self.net is None or T <= self.lookback:
            return out
        feats = df[self.feature_cols].values.astype(np.float32)
        feats = (feats - self.feat_mean) / self.feat_std
        N = T - self.lookback
        X = np.stack([feats[i:i+self.lookback] for i in range(N)], axis=0)
        Xt = th.from_numpy(X).to(self.device)
        self.net.eval()
        with th.no_grad():
            preds = self.net(Xt).cpu().numpy()
        out[self.lookback:self.lookback+N, 0] = preds
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost Forecaster — tabular / non-linear interactions
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostForecaster(BaseForecaster):
    name = "xgboost"

    def __init__(
        self,
        feature_cols: list,
        lookback: int = 1,            # XGB uses tabular features at time t (no sequence)
        n_estimators: int = 200,
        max_depth: int    = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.feature_cols  = list(feature_cols)
        self.lookback      = lookback   # 1 = use only current-day features
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model         = None
        self.sigma         = 0.02

    def fit(self, df_train: pd.DataFrame) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        # Predict next-day return from current-day features
        X = df_train[self.feature_cols].values[:-1]   # features at t
        y = df_train["log_return"].fillna(0).values[1:]  # target = log_ret at t+1
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self

        self.model = XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, objective="reg:squarederror",
            random_state=self.random_state, n_jobs=-1, verbosity=0,
        )
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.sigma = float(np.std(y - preds) + 1e-6)
        return self

    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        out = np.zeros((T, 2), dtype=np.float32)
        out[:, 1] = self.sigma
        if self.model is None:
            return out
        X = df[self.feature_cols].values
        # Predict next-day return for each row; shift by 1 so out[t] = E[r_{t+1}|x_t]
        preds = self.model.predict(X).astype(np.float32)
        out[:T-1, 0] = preds[:T-1]
        out[T-1, 0]  = 0.0  # last row has no next-day target
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. CNN Forecaster — local pattern extraction
# ─────────────────────────────────────────────────────────────────────────────

class _CNNNet(nn.Module):
    def __init__(self, n_features, channels=32, lookback=60):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, T, F) → permute to (B, F, T) for Conv1d
        z = x.permute(0, 2, 1)
        z = self.cnn(z).squeeze(-1)
        return self.head(z).squeeze(-1)


class CNNForecaster(BaseForecaster):
    name = "cnn"

    def __init__(
        self,
        feature_cols: list,
        lookback: int = 60,
        channels: int = 32,
        epochs: int   = 15,
        batch_size: int = 64,
        lr: float     = 1e-3,
        device: str   = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.lookback     = lookback
        self.channels     = channels
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.device       = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else device if device != "auto" else "cpu")
        self.net          = None
        self.feat_mean    = None
        self.feat_std     = None
        self.sigma        = 0.02

    def fit(self, df_train: pd.DataFrame) -> "CNNForecaster":
        feats = df_train[self.feature_cols].values.astype(np.float32)
        self.feat_mean = feats.mean(0, keepdims=True)
        self.feat_std  = feats.std(0, keepdims=True) + 1e-8

        X, y = _build_sequences(df_train, self.feature_cols, self.lookback)
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self
        X = (X - self.feat_mean) / self.feat_std
        Xt = th.from_numpy(X).to(self.device)
        yt = th.from_numpy(y).to(self.device)

        self.net = _CNNNet(len(self.feature_cols), self.channels, self.lookback).to(self.device)
        opt = th.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.mse_loss(self.net(xb), yb)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

        self.net.eval()
        with th.no_grad():
            preds = self.net(Xt).cpu().numpy()
        self.sigma = float(np.std(y - preds) + 1e-6)
        return self

    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        out = np.zeros((T, 2), dtype=np.float32)
        out[:, 1] = self.sigma
        if self.net is None or T <= self.lookback:
            return out
        feats = df[self.feature_cols].values.astype(np.float32)
        feats = (feats - self.feat_mean) / self.feat_std
        N = T - self.lookback
        X = np.stack([feats[i:i+self.lookback] for i in range(N)], axis=0)
        Xt = th.from_numpy(X).to(self.device)
        self.net.eval()
        with th.no_grad():
            preds = self.net(Xt).cpu().numpy()
        out[self.lookback:self.lookback+N, 0] = preds
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. GARCH Forecaster — conditional volatility
# ─────────────────────────────────────────────────────────────────────────────

class GARCHForecaster(BaseForecaster):
    """
    GARCH(1,1) for σ_{t+1} prediction.  μ is set to the in-sample mean return
    (very close to 0 for daily data); the value here is the σ forecast.

    Reference: Bollerslev (1986) "Generalised autoregressive conditional
    heteroskedasticity", Journal of Econometrics 31:307-327.
    """
    name = "garch"

    def __init__(self, scale: float = 100.0) -> None:
        # arch package complains about scale; we work in percent then rescale
        self.scale  = scale
        self.fit_result = None
        self.mu     = 0.0
        self.sigma_const = 0.02

    def fit(self, df_train: pd.DataFrame) -> "GARCHForecaster":
        from arch import arch_model
        rets = (df_train["log_return"].fillna(0) * self.scale).values
        if len(rets) < 100:
            self.sigma_const = float(df_train["log_return"].std())
            self.mu          = float(df_train["log_return"].mean())
            return self
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(rets, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
                self.fit_result = model.fit(disp="off", show_warning=False)
            self.mu = float(df_train["log_return"].mean())
        except Exception:
            self.fit_result = None
            self.sigma_const = float(df_train["log_return"].std())
            self.mu          = float(df_train["log_return"].mean())
        return self

    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        out = np.zeros((T, 2), dtype=np.float32)
        out[:, 0] = self.mu

        if self.fit_result is None:
            out[:, 1] = self.sigma_const
            return out

        # Refit on test data's cumulative returns to get conditional vol path
        rets = (df["log_return"].fillna(0) * self.scale).values
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use fitted parameters; recompute conditional volatility on new data
                params = self.fit_result.params
                model  = arch_model(rets, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
                fit2   = model.fix(params)
                cv     = fit2.conditional_volatility / self.scale
            out[:, 1] = cv.astype(np.float32)
        except Exception:
            out[:, 1] = self.sigma_const
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory: build all 5 forecasters with sensible defaults
# ─────────────────────────────────────────────────────────────────────────────

def build_default_forecasters(
    feature_cols: list,
    lookback: int = 60,
    epochs: int   = 15,
    device: str   = "auto",
) -> list:
    """Return a list of 5 BaseForecaster instances with sensible defaults."""
    return [
        LSTMForecaster(feature_cols, lookback=lookback, epochs=epochs, device=device),
        TransformerForecaster(feature_cols, lookback=lookback, epochs=epochs, device=device),
        XGBoostForecaster(feature_cols),
        CNNForecaster(feature_cols, lookback=lookback, epochs=epochs, device=device),
        GARCHForecaster(),
    ]
