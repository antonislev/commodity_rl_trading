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
    """All forecasters predict next-day log-return (μ) and its std (σ).

    v3.3: fit() now accepts optional `sample_weights` for regime-specialised
    training.  Weights have shape (n_samples,) and are aligned to df_train
    BEFORE the lookback windowing.  Downstream forecasters use them in
    their loss (weighted MSE for the deep nets, per-sample weights for
    XGBoost).  Pass None to fall back to uniform weighting (default).
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        df_train: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "BaseForecaster": ...

    @abstractmethod
    def predict_mu_sigma(self, df: pd.DataFrame) -> np.ndarray: ...


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
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


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
        dropout: float = 0.3,            # v3.2: stronger regularisation
        val_split: float = 0.2,          # v3.2: chronological tail used as validation
        patience: int  = 3,              # v3.2: early-stopping patience
        weight_decay: float = 1e-5,      # v3.2: L2 regularisation
        device: str   = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.lookback     = lookback
        self.hidden       = hidden
        self.n_layers     = n_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.dropout      = dropout
        self.val_split    = val_split
        self.patience     = patience
        self.weight_decay = weight_decay
        self.device       = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else device if device != "auto" else "cpu")
        self.net: Optional[_LSTMNet]            = None
        self.feat_mean: Optional[np.ndarray]    = None
        self.feat_std:  Optional[np.ndarray]    = None
        self.sigma:     float                   = 0.02

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / (self.feat_std + 1e-8)

    def fit(
        self,
        df_train: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "LSTMForecaster":
        feats = df_train[self.feature_cols].values.astype(np.float32)
        self.feat_mean = feats.mean(0, keepdims=True)
        self.feat_std  = feats.std(0, keepdims=True) + 1e-8

        X, y = _build_sequences(df_train, self.feature_cols, self.lookback)
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self
        X = (X - self.feat_mean) / self.feat_std

        # v3.3: sample weights aligned with target indices (i.e. position lookback+i)
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float32)
            w = w[self.lookback:self.lookback + len(X)] if len(w) > len(X) else w[:len(X)]
            # Normalise to mean=1 so loss scale is unchanged
            w = w * (len(w) / (w.sum() + 1e-8))
        else:
            w = np.ones(len(X), dtype=np.float32)

        n_val = max(int(len(X) * self.val_split), 10)
        X_tr, X_vl = X[:-n_val], X[-n_val:]
        y_tr, y_vl = y[:-n_val], y[-n_val:]
        w_tr        = w[:-n_val]
        Xtr_t = th.from_numpy(X_tr).to(self.device)
        ytr_t = th.from_numpy(y_tr).to(self.device)
        wtr_t = th.from_numpy(w_tr).to(self.device)
        Xvl_t = th.from_numpy(X_vl).to(self.device)
        yvl_t = th.from_numpy(y_vl).to(self.device)

        self.net = _LSTMNet(len(self.feature_cols), self.hidden, self.n_layers, self.dropout).to(self.device)
        opt = th.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t, wtr_t), batch_size=self.batch_size, shuffle=True,
        )

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for _ in range(self.epochs):
            self.net.train()
            for xb, yb, wb in loader:
                opt.zero_grad()
                pred = self.net(xb)
                # Weighted MSE
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            self.net.eval()
            with th.no_grad():
                val_loss = F.mse_loss(self.net(Xvl_t), yvl_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        with th.no_grad():
            preds_full = self.net(th.from_numpy(X).to(self.device)).cpu().numpy()
        self.sigma = float(np.std(y - preds_full) + 1e-6)
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
    def __init__(self, n_features, d_model=32, nhead=4, n_layers=2, lookback=60, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(th.randn(1, lookback, d_model) * 0.02)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_drop = nn.Dropout(dropout)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
        z = self.encoder(z)
        return self.head(self.head_drop(z[:, -1, :])).squeeze(-1)


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
        dropout: float = 0.3,
        val_split: float = 0.2,
        patience: int  = 3,
        weight_decay: float = 1e-5,
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
        self.dropout      = dropout
        self.val_split    = val_split
        self.patience     = patience
        self.weight_decay = weight_decay
        self.device       = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else device if device != "auto" else "cpu")
        self.net: Optional[_TransformerNet] = None
        self.feat_mean = None
        self.feat_std  = None
        self.sigma     = 0.02

    def fit(
        self,
        df_train: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "TransformerForecaster":
        feats = df_train[self.feature_cols].values.astype(np.float32)
        self.feat_mean = feats.mean(0, keepdims=True)
        self.feat_std  = feats.std(0, keepdims=True) + 1e-8

        X, y = _build_sequences(df_train, self.feature_cols, self.lookback)
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self
        X = (X - self.feat_mean) / self.feat_std

        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float32)
            w = w[self.lookback:self.lookback + len(X)] if len(w) > len(X) else w[:len(X)]
            w = w * (len(w) / (w.sum() + 1e-8))
        else:
            w = np.ones(len(X), dtype=np.float32)

        n_val = max(int(len(X) * self.val_split), 10)
        X_tr, X_vl = X[:-n_val], X[-n_val:]
        y_tr, y_vl = y[:-n_val], y[-n_val:]
        w_tr = w[:-n_val]
        Xtr_t = th.from_numpy(X_tr).to(self.device)
        ytr_t = th.from_numpy(y_tr).to(self.device)
        wtr_t = th.from_numpy(w_tr).to(self.device)
        Xvl_t = th.from_numpy(X_vl).to(self.device)
        yvl_t = th.from_numpy(y_vl).to(self.device)

        self.net = _TransformerNet(
            len(self.feature_cols), self.d_model, self.nhead, self.n_layers, self.lookback,
            dropout=self.dropout,
        ).to(self.device)
        opt = th.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t, wtr_t), batch_size=self.batch_size, shuffle=True,
        )

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for _ in range(self.epochs):
            self.net.train()
            for xb, yb, wb in loader:
                opt.zero_grad()
                pred = self.net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with th.no_grad():
                val_loss = F.mse_loss(self.net(Xvl_t), yvl_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        with th.no_grad():
            preds_full = self.net(th.from_numpy(X).to(self.device)).cpu().numpy()
        self.sigma = float(np.std(y - preds_full) + 1e-6)
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

    def fit(
        self,
        df_train: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        X = df_train[self.feature_cols].values[:-1]
        y = df_train["log_return"].fillna(0).values[1:]
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self

        # v3.3 regime-specialised: align weights with X (drop last entry)
        sw = None
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float32)
            sw = sw[:len(X)]
            sw = sw * (len(sw) / (sw.sum() + 1e-8))

        self.model = XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, objective="reg:squarederror",
            random_state=self.random_state, n_jobs=-1, verbosity=0,
        )
        self.model.fit(X, y, sample_weight=sw)
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
        val_split: float = 0.2,
        patience: int  = 3,
        weight_decay: float = 1e-5,
        device: str   = "auto",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.lookback     = lookback
        self.channels     = channels
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.val_split    = val_split
        self.patience     = patience
        self.weight_decay = weight_decay
        self.device       = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else device if device != "auto" else "cpu")
        self.net          = None
        self.feat_mean    = None
        self.feat_std     = None
        self.sigma        = 0.02

    def fit(
        self,
        df_train: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "CNNForecaster":
        feats = df_train[self.feature_cols].values.astype(np.float32)
        self.feat_mean = feats.mean(0, keepdims=True)
        self.feat_std  = feats.std(0, keepdims=True) + 1e-8

        X, y = _build_sequences(df_train, self.feature_cols, self.lookback)
        if len(X) < 50:
            self.sigma = float(df_train["log_return"].std())
            return self
        X = (X - self.feat_mean) / self.feat_std

        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float32)
            w = w[self.lookback:self.lookback + len(X)] if len(w) > len(X) else w[:len(X)]
            w = w * (len(w) / (w.sum() + 1e-8))
        else:
            w = np.ones(len(X), dtype=np.float32)

        n_val = max(int(len(X) * self.val_split), 10)
        X_tr, X_vl = X[:-n_val], X[-n_val:]
        y_tr, y_vl = y[:-n_val], y[-n_val:]
        w_tr = w[:-n_val]
        Xtr_t = th.from_numpy(X_tr).to(self.device)
        ytr_t = th.from_numpy(y_tr).to(self.device)
        wtr_t = th.from_numpy(w_tr).to(self.device)
        Xvl_t = th.from_numpy(X_vl).to(self.device)
        yvl_t = th.from_numpy(y_vl).to(self.device)

        self.net = _CNNNet(len(self.feature_cols), self.channels, self.lookback).to(self.device)
        opt = th.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t, wtr_t), batch_size=self.batch_size, shuffle=True,
        )

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for _ in range(self.epochs):
            self.net.train()
            for xb, yb, wb in loader:
                opt.zero_grad()
                pred = self.net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with th.no_grad():
                val_loss = F.mse_loss(self.net(Xvl_t), yvl_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        with th.no_grad():
            preds_full = self.net(th.from_numpy(X).to(self.device)).cpu().numpy()
        self.sigma = float(np.std(y - preds_full) + 1e-6)
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

    def fit(
        self,
        df_train: pd.DataFrame,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "GARCHForecaster":
        # GARCH does not use sample_weights (volatility model)
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
