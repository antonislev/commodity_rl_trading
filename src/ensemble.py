"""
ARA-PPO v3 — Regime-Aware Ensemble Combiner
=============================================
Combines per-forecaster (μ, σ) predictions into a single ensemble forecast,
where the per-forecaster weights depend on the current HMM regime.

Idea
----
Different forecasters excel in different market regimes:
  - LSTM         strong in trending periods (autoregressive momentum)
  - Transformer  strong with long-range dependencies
  - XGBoost      strong with non-linear feature interactions in stable regimes
  - CNN          strong with local price patterns (breakouts)
  - GARCH        strong σ estimates in volatile periods

A single fixed weighting cannot exploit this.  We learn a separate weight
vector per regime and blend at prediction time using the HMM regime
probabilities.

Math
----
Given:
  μ_i, σ_i      : prediction from forecaster i ∈ {1..K}
  p_r           : HMM probability of regime r ∈ {1..R}
  W_{r,i}       : learnable weight for regime r and forecaster i

Effective per-timestep forecaster weights:
  w_i = Σ_r p_r W_{r,i}                       (mixture over regimes)
  w_i ← w_i / Σ_j w_j                          (normalise to simplex)

Ensemble prediction:
  μ_ens = Σ_i w_i μ_i
  σ_ens = sqrt( Σ_i w_i σ_i²  +  Σ_i w_i (μ_i − μ_ens)² )
            └─── within-forecaster ──┘  └─ between-forecaster ─┘

The latter is the law of total variance: variance of a mixture =
expected within-component variance + variance of component means.

Weights W are fit by minimising MSE of μ_ens against realised next-day
returns on the training data, conditional on the HMM regime distribution.

This is closed-form per regime if we ignore the simplex constraint, but
since we want non-negative weights we use a softmax parameterisation and
gradient descent on a tiny problem (R × K params, typically 4 × 5 = 20).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RegimeAwareEnsemble:
    """
    Per-regime softmax weighting of forecasters.

    Parameters
    ----------
    n_regimes : int
        Number of HMM regimes
    n_forecasters : int
        Number of base forecasters
    n_iter : int
        Gradient-descent iterations for weight fitting
    lr : float
        Learning rate for weight fitting
    """

    def __init__(
        self,
        n_regimes: int     = 4,
        n_forecasters: int = 5,
        n_iter: int        = 1000,
        lr: float          = 0.05,
        # v3.3: adaptive softmax temperature
        # final temp = base_temp + alpha_temp * uncertainty
        base_temp: float        = 1.0,
        alpha_temp: float       = 1.5,
    ) -> None:
        self.n_regimes     = n_regimes
        self.n_forecasters = n_forecasters
        self.n_iter        = n_iter
        self.lr            = lr
        self.base_temp     = base_temp
        self.alpha_temp    = alpha_temp
        self.logits: Optional[th.Tensor] = None
        self._fitted: bool = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        forecaster_mus:    np.ndarray,     # (T, K)
        regime_probs:      np.ndarray,     # (T, R)
        target_returns:    np.ndarray,     # (T,) realised log_return shifted by +1
    ) -> "RegimeAwareEnsemble":
        """
        Fit per-regime forecaster weights by minimising MSE on training data.

          loss = mean_t (Σ_i w_i(t) μ_i(t) − y(t))²

        where w_i(t) = softmax over i of (regime_probs(t) @ logits)_i.
        """
        # Drop initial rows where forecasters have no real prediction (μ=0)
        mask = ~np.isclose(forecaster_mus, 0.0, atol=1e-12).all(axis=1)
        # Also drop rows with NaN regime probs or target
        mask &= ~np.isnan(regime_probs).any(axis=1)
        mask &= ~np.isnan(target_returns)

        if mask.sum() < 30:
            # Not enough data — fall back to equal weights
            self.logits = th.zeros(self.n_regimes, self.n_forecasters)
            self._fitted = True
            return self

        mu_t = th.from_numpy(forecaster_mus[mask].astype(np.float32))
        rp_t = th.from_numpy(regime_probs[mask].astype(np.float32))
        y_t  = th.from_numpy(target_returns[mask].astype(np.float32))

        logits = th.zeros(self.n_regimes, self.n_forecasters, requires_grad=True)
        opt = th.optim.Adam([logits], lr=self.lr)

        for _ in range(self.n_iter):
            opt.zero_grad()
            # weights[t, i] = softmax_i ( Σ_r rp_t[t,r] * logits[r, i] )
            mixed_logits = rp_t @ logits                          # (T, K)
            w = F.softmax(mixed_logits, dim=-1)                    # (T, K)
            mu_ens = (w * mu_t).sum(dim=-1)                        # (T,)
            loss = F.mse_loss(mu_ens, y_t) + 1e-4 * (logits ** 2).sum()
            loss.backward()
            opt.step()

        self.logits = logits.detach().clone()
        self._fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        forecaster_mus:    np.ndarray,    # (T, K)
        forecaster_sigmas: np.ndarray,    # (T, K)
        regime_probs:      np.ndarray,    # (T, R)
    ) -> np.ndarray:
        """
        Return (T, 2) array — column 0 is ensemble μ, column 1 is ensemble σ.
        v3.3: uses adaptive softmax temperature scaled by forecaster
        disagreement.  When forecasters disagree strongly (high uncertainty)
        the temperature increases, producing softer weights (closer to
        uniform).  When they agree, weights sharpen toward the best-trusted
        forecaster for the current regime.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        rp = th.from_numpy(regime_probs.astype(np.float32))
        mixed_logits = (rp @ self.logits).numpy()                  # (T, K)

        mu_i    = forecaster_mus.astype(np.float32)
        sigma_i = forecaster_sigmas.astype(np.float32)

        # v3.3 adaptive temperature: per-timestep disagreement
        disagreement = mu_i.std(axis=-1, keepdims=True)            # (T, 1)
        # Normalise by median for scale invariance
        scale = float(np.median(disagreement) + 1e-8)
        tau = self.base_temp + self.alpha_temp * (disagreement / scale).squeeze(-1)
        tau = np.clip(tau, 0.5, 5.0)                                # (T,)

        # Apply per-row temperature
        exp_logits = np.exp(mixed_logits / tau[:, None])
        w = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-12)

        mu_ens  = (w * mu_i).sum(axis=-1)
        var_within  = (w * sigma_i**2).sum(axis=-1)
        var_between = (w * (mu_i - mu_ens[:, None])**2).sum(axis=-1)
        sigma_ens = np.sqrt(np.maximum(var_within + var_between, 1e-12))

        out = np.zeros((len(mu_ens), 2), dtype=np.float32)
        out[:, 0] = mu_ens
        out[:, 1] = sigma_ens
        return out

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def regime_weight_matrix(self) -> np.ndarray:
        """Return (n_regimes, n_forecasters) softmaxed weights — interpretable."""
        if self.logits is None:
            return np.full((self.n_regimes, self.n_forecasters), 1.0 / self.n_forecasters)
        return F.softmax(self.logits, dim=-1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# v3.3 — regime classifier (trend / volatile / crisis)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regimes(hmm) -> dict:
    """
    Map each HMM regime index to a semantic label and a hard Kelly cap.

    Labels assigned from regime characteristics:
      crisis    : high σ AND negative smoothed μ
      volatile  : high σ
      trend     : low σ

    Returns
    -------
    dict with:
      'labels'        : list[str] of length n_regimes
      'kelly_caps'    : np.ndarray of length n_regimes
      'sample_weights': np.ndarray (n_regimes, n_regimes) — for regime-specialised
                        forecaster training.  Row i = weights for forecaster
                        specialised in regime i.
    """
    means = hmm.regime_means()                                # (R, F)
    n_regimes = means.shape[0]
    sigmas    = means[:, 1] if means.shape[1] >= 2 else np.full(n_regimes, 0.02)
    smoothed_mu = means[:, 2] if means.shape[1] >= 3 else np.zeros(n_regimes)

    # Vol classification: split into "high" (above median) vs "low"
    sigma_median = float(np.median(sigmas))
    labels: list = []
    kelly_caps  = np.zeros(n_regimes, dtype=np.float32)
    for k in range(n_regimes):
        is_high_vol = sigmas[k] > sigma_median * 1.3
        is_crisis   = is_high_vol and smoothed_mu[k] < -0.003
        if is_crisis:
            labels.append("crisis")
            kelly_caps[k] = 0.05
        elif is_high_vol:
            labels.append("volatile")
            kelly_caps[k] = 0.20
        else:
            labels.append("trend")
            kelly_caps[k] = 0.50

    return {
        "labels":     labels,
        "kelly_caps": kelly_caps,
        "n_regimes":  n_regimes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# v3.1 — regime-aware Kelly schedule (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def kelly_per_regime_from_hmm(
    hmm,
    base_kelly: float       = 0.5,
    crisis_threshold: float = -0.005,
    crisis_penalty: float   = 0.5,
    kelly_min: float        = 0.10,
    kelly_max: float        = 1.0,
) -> np.ndarray:
    """
    Derive a per-regime Kelly fraction from HMM regime characteristics.

    Logic
    -----
    Each regime has a mean (μ, σ, smoothed_μ) tuple from the fitted HMM.
    We map regime σ to Kelly fraction inversely (low-vol regimes get more
    leverage), and apply an extra penalty for "crisis-like" regimes
    (those with strongly negative smoothed mean return).

      kelly_r  =  base_kelly * (median_σ / σ_r)
      kelly_r *= 1 if smoothed_μ_r > -threshold else crisis_penalty
      kelly_r  = clip(kelly_r, kelly_min, kelly_max)

    Returns
    -------
    np.ndarray shape (n_regimes,) with per-regime Kelly fractions.
    """
    means = hmm.regime_means()                          # (n_regimes, n_feat)
    if means.shape[1] < 2:
        # Degenerate HMM — return uniform base_kelly
        return np.full(means.shape[0], base_kelly, dtype=np.float32)

    regime_vol = np.maximum(means[:, 1], 1e-4)
    median_vol = float(np.median(regime_vol))
    kelly_per  = base_kelly * (median_vol / regime_vol)

    # Crisis penalty: regimes with strongly negative mean return get reduced kelly
    if means.shape[1] >= 3:
        smoothed_mu = means[:, 2]
        is_crisis   = smoothed_mu < crisis_threshold
        kelly_per   = np.where(is_crisis, kelly_per * crisis_penalty, kelly_per)

    return np.clip(kelly_per, kelly_min, kelly_max).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: stack per-forecaster predictions on a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def stack_forecaster_predictions(
    forecasters: Sequence,
    df: pd.DataFrame,
) -> tuple:
    """
    Run all forecasters on df and stack their μ and σ predictions.

    Returns (mus, sigmas) both shape (T, K).
    """
    mus = []
    sigs = []
    for fc in forecasters:
        ms = fc.predict_mu_sigma(df)
        mus.append(ms[:, 0])
        sigs.append(ms[:, 1])
    return (np.column_stack(mus).astype(np.float32),
            np.column_stack(sigs).astype(np.float32))
