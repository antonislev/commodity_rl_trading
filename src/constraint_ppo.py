"""
ARA-PPO v3 — Constraint-Aware PPO with compact ensemble-driven state
======================================================================
Components
----------
  HybridForecastEnv         wraps CommodityTradingEnv; replaces the 2825-dim
                            observation with a compact 12-dim state:
                            [μ_ens, σ_ens, sharpe_estimate, regime_probs(4),
                             portfolio_state(5)]

  ConstraintAwarePolicy     small MLP actor-critic; the action head is
                            constrained by a Kelly fraction so position size
                            cannot exceed |μ/σ²| (capped at 1.0)

  make_compact_ppo          factory that builds PPO on the compact state space

Why a compact state space?
--------------------------
The v2 PPO was given a 2825-dim observation (60-day market window).  PPO's
policy gradient signal-to-noise ratio is extremely poor on financial data
(~1/125), so it cannot effectively learn to forecast from raw features.  The
ensemble does the forecasting via supervised learning (much more sample-
efficient).  PPO then only needs to decide: given a forecast (μ, σ), what
position to take.  This is a much simpler problem, learnable by a small MLP.

Why constraint-aware?
---------------------
Kelly criterion gives the optimal position f* = μ/σ².  Without a constraint,
PPO can over-leverage on noisy forecasts.  We bound |action| ≤ |Kelly| so
the policy can only choose direction confidence and a fraction of the Kelly
optimum, never exceed it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid environment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HybridForecastEnv(gym.Env):
    """
    Wraps a `CommodityTradingEnv` and exposes a compact observation built from
    the regime-aware ensemble forecast plus minimal portfolio state.

    obs (12 dims)
    -------------
      [0]   ensemble μ        — predicted next-day log-return
      [1]   ensemble σ        — predicted next-day log-return std
      [2]   Sharpe estimate   — μ / σ, clipped to [-5, +5]
      [3:7] HMM regime probs  — (P(r_0), P(r_1), P(r_2), P(r_3))
      [7]   current position
      [8]   drawdown ∈ [-1, 0]
      [9]   log_pnl
      [10]  episode progress ∈ [0, 1]
      [11]  cost ratio ∈ [0, 0.1]

    Action: scalar ∈ [-1, +1] (target position).  The base env handles the
    transition cost, position-change limit, etc.

    Constructor parameters
    ----------------------
    base_env : CommodityTradingEnv   (already-built per-split env)
    forecast_mus    : (T, K)         per-forecaster μ predictions on this env's data
    forecast_sigmas : (T, K)         per-forecaster σ predictions on this env's data
    regime_probs    : (T, R)         HMM regime probabilities on this env's data
    ensemble        : RegimeAwareEnsemble  — fitted ensemble combiner

    Note
    ----
    All four arrays must have length == len(base_env.df).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_env,
        forecast_mus:    np.ndarray,
        forecast_sigmas: np.ndarray,
        regime_probs:    np.ndarray,
        ensemble,
        # v3.1 risk-aware parameters
        sigma_garch_idx:    int          = -1,
        sigma_train_floor:  float        = 0.0,
        target_vol:         Optional[float] = None,
        vol_window:         int          = 20,
        vol_scaler_min:     float        = 0.3,
        vol_scaler_max:     float        = 1.5,
        # v3.2 OOD + forecast-quality feedback
        use_ood_attenuation: bool        = False,
        ood_threshold:       float       = 0.02,    # legacy; preferred is ood_p90 below
        ood_p90:             Optional[float] = None, # v3.3: 90th-percentile training disagreement
        ood_attenuation_alpha: float     = 0.5,
        use_ewma_corr:       bool        = False,
        ewma_window:         int         = 60,      # v3.3: longer window (was 20)
        # v3.3 new features
        use_autocorr_feature: bool       = False,
        autocorr_window:      int        = 20,
        use_disagreement_in_sigma: bool  = False,
        use_rolling_vol_in_sigma:  bool  = False,
        rolling_vol_window:        int    = 20,
    ) -> None:
        super().__init__()
        self.base_env = base_env
        self.forecast_mus    = forecast_mus.astype(np.float32)
        self.forecast_sigmas = forecast_sigmas.astype(np.float32)
        self.regime_probs    = regime_probs.astype(np.float32)
        self.ensemble        = ensemble

        # v3.1 — σ safety params
        self.sigma_garch_idx    = int(sigma_garch_idx)
        self.sigma_train_floor  = float(sigma_train_floor)
        # v3.1 — vol-targeting params
        self.target_vol         = float(target_vol) if target_vol is not None else None
        self.vol_window         = int(vol_window)
        self.vol_scaler_min     = float(vol_scaler_min)
        self.vol_scaler_max     = float(vol_scaler_max)
        # v3.2 / v3.3 — OOD + EWMA correlation params
        self.use_ood_attenuation   = bool(use_ood_attenuation)
        self.ood_threshold         = float(ood_threshold)
        self.ood_p90               = ood_p90    # if set, used in preference to ood_threshold
        self.ood_attenuation_alpha = float(ood_attenuation_alpha)
        self.use_ewma_corr         = bool(use_ewma_corr)
        self.ewma_window           = int(ewma_window)
        self._ewma_alpha           = 2.0 / (max(self.ewma_window, 2) + 1)

        # v3.3 — new features
        self.use_autocorr_feature       = bool(use_autocorr_feature)
        self.autocorr_window            = int(autocorr_window)
        self.use_disagreement_in_sigma  = bool(use_disagreement_in_sigma)
        self.use_rolling_vol_in_sigma   = bool(use_rolling_vol_in_sigma)
        self.rolling_vol_window         = int(rolling_vol_window)

        # Pre-compute ensemble forecasts for all timesteps
        self._ensemble_pred = ensemble.predict(
            forecast_mus, forecast_sigmas, regime_probs,
        )  # (T, 2)

        T = len(self.forecast_mus)
        if T != len(base_env.df):
            raise ValueError(
                f"Forecast length {T} != base env data length {len(base_env.df)}"
            )

        n_regimes = self.regime_probs.shape[1]
        # v3.3 obs layout (extras placed before regime, in this fixed order):
        #   ood, recent_corr (reliability), autocorr
        self._n_extra = (
            (1 if self.use_ood_attenuation else 0)
            + (1 if self.use_ewma_corr else 0)
            + (1 if self.use_autocorr_feature else 0)
        )
        self._obs_dim = 3 + self._n_extra + n_regimes + 5
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = base_env.action_space

        # v3.1 — diagnostics
        self._last_sigma_eff:    Optional[float] = None
        self._last_vol_scaler:   Optional[float] = None
        self._last_kelly_frac:   Optional[float] = None
        # v3.2 — diagnostics + EWMA state
        self._last_ood_score:    Optional[float] = None
        self._last_recent_corr:  Optional[float] = None
        self._last_autocorr:     Optional[float] = None
        self._reset_ewma_state()
        self._pending_pred_mu:   Optional[float] = None

    def _reset_ewma_state(self) -> None:
        self._ewma_mu  = 0.0
        self._ewma_r   = 0.0
        self._ewma_mu2 = 0.0
        self._ewma_r2  = 0.0
        self._ewma_mur = 0.0
        self._ewma_count = 0

    def _update_ewma(self, mu: float, r: float) -> None:
        a = self._ewma_alpha
        self._ewma_mu  = a * mu      + (1 - a) * self._ewma_mu
        self._ewma_r   = a * r       + (1 - a) * self._ewma_r
        self._ewma_mu2 = a * mu * mu + (1 - a) * self._ewma_mu2
        self._ewma_r2  = a * r  * r  + (1 - a) * self._ewma_r2
        self._ewma_mur = a * mu * r  + (1 - a) * self._ewma_mur
        self._ewma_count += 1

    def _ewma_correlation(self) -> float:
        if self._ewma_count < 5:
            return 0.0
        var_mu = self._ewma_mu2 - self._ewma_mu ** 2
        var_r  = self._ewma_r2  - self._ewma_r ** 2
        if var_mu <= 1e-10 or var_r <= 1e-10:
            return 0.0
        cov = self._ewma_mur - self._ewma_mu * self._ewma_r
        c = cov / (np.sqrt(var_mu * var_r) + 1e-10)
        return float(np.clip(c, -1.0, 1.0))

    def _compute_ood_score(self, idx: int) -> float:
        """
        v3.3 OOD: composite score from forecaster disagreement + reliability decay.

          ood_disagreement = sigmoid((d - p90) / p90)   sharp activation above 90th-pct
          ood_reliability  = max(0, -recent_corr)        positive when forecasts wrong-direction
          ood = 0.6 * disagreement + 0.4 * reliability

        With ood_p90 set, the disagreement component only fires when disagreement
        exceeds the 90th percentile of training-set disagreement — addressing
        the v3.2 bug where OOD fired constantly.
        """
        mus_t = self.forecast_mus[idx]
        disagreement = float(np.std(mus_t))

        if self.ood_p90 is not None and self.ood_p90 > 0:
            # Sigmoid centred at p90 — fires sharply only on outlier disagreement
            x = (disagreement - self.ood_p90) / max(self.ood_p90, 1e-6)
            ood_dis = float(1.0 / (1.0 + np.exp(-3.0 * x)))   # logistic
        else:
            # Legacy fallback
            ood_dis = float(1.0 - np.exp(-disagreement / max(self.ood_threshold, 1e-6)))

        # Reliability component: when EWMA corr goes negative, forecasts are
        # systematically wrong-direction → treat as OOD-like signal
        if self.use_ewma_corr:
            recent_corr = self._ewma_correlation()
            ood_rel = max(0.0, -recent_corr)                  # in [0, 1]
        else:
            ood_rel = 0.0

        return float(np.clip(0.6 * ood_dis + 0.4 * ood_rel, 0.0, 1.0))

    def _compute_autocorr(self) -> float:
        """v3.3 chop detector: lag-1 autocorrelation of recent realised returns.
        Positive autocorr = trending; near zero or negative = choppy/mean-reverting."""
        env = self.base_env
        idx = env.start_idx + env.lookback + env.current_step
        # Use realised market returns history (from env's return_arr)
        lo = max(0, idx - self.autocorr_window)
        rets = env.return_arr[lo:idx]
        if len(rets) < 5:
            return 0.0
        try:
            r = np.asarray(rets, dtype=np.float64)
            r1 = r[1:]
            r0 = r[:-1]
            if r1.std() < 1e-8 or r0.std() < 1e-8:
                return 0.0
            corr = float(np.corrcoef(r0, r1)[0, 1])
            if not np.isfinite(corr):
                return 0.0
            return float(np.clip(corr, -1.0, 1.0))
        except Exception:
            return 0.0

    def _rolling_realised_vol(self) -> float:
        """v3.3 σ floor from recent realised market vol (annualised)."""
        env = self.base_env
        idx = env.start_idx + env.lookback + env.current_step
        lo = max(0, idx - self.rolling_vol_window)
        rets = env.return_arr[lo:idx]
        if len(rets) < 5:
            return 0.0
        v = float(np.std(rets) * np.sqrt(252))
        # Convert annualised back to daily-scale σ
        return v / np.sqrt(252)

    # ── Compact observation builder ───────────────────────────────────────────

    def _compact_obs(self) -> np.ndarray:
        idx = self.base_env.start_idx + self.base_env.lookback + self.base_env.current_step
        idx = min(idx, len(self._ensemble_pred) - 1)

        mu      = float(self._ensemble_pred[idx, 0])
        sig_ens = float(max(self._ensemble_pred[idx, 1], 1e-6))

        # ── v3.3 σ_final = max(σ_ens, σ_garch, σ_train_floor, disagreement, rolling_vol) ──
        sig_eff = sig_ens
        if 0 <= self.sigma_garch_idx < self.forecast_sigmas.shape[1]:
            sig_garch = float(self.forecast_sigmas[idx, self.sigma_garch_idx])
            if sig_garch > 0 and np.isfinite(sig_garch):
                sig_eff = max(sig_eff, sig_garch)
        if self.sigma_train_floor > 0:
            sig_eff = max(sig_eff, self.sigma_train_floor)
        # v3.3 new components
        if self.use_disagreement_in_sigma:
            disagreement = float(np.std(self.forecast_mus[idx]))
            sig_eff = max(sig_eff, disagreement)
        if self.use_rolling_vol_in_sigma:
            sig_eff = max(sig_eff, self._rolling_realised_vol())

        self._last_sigma_eff = sig_eff
        sharpe = float(np.clip(mu / sig_eff, -5.0, 5.0))
        regime = self.regime_probs[idx]                       # (R,)

        ood_score   = self._compute_ood_score(idx) if self.use_ood_attenuation else 0.0
        recent_corr = self._ewma_correlation() if self.use_ewma_corr else 0.0
        autocorr    = self._compute_autocorr() if self.use_autocorr_feature else 0.0
        self._last_ood_score   = ood_score
        self._last_recent_corr = recent_corr
        self._last_autocorr    = autocorr

        env = self.base_env
        peak = max(env.peak_value, 1e-9)
        drawdown = env.portfolio_value / peak - 1.0 if peak > 0 else 0.0
        log_pnl  = float(np.log(env.portfolio_value / env.config["initial_capital"] + 1e-10))
        portfolio = np.array([
            env.position,
            np.clip(drawdown, -1.0, 0.0),
            np.clip(log_pnl, -5.0, 5.0),
            env.current_step / max(env.episode_length, 1),
            np.clip(env.total_costs / env.config["initial_capital"], 0, 0.1),
        ], dtype=np.float32)

        self._pending_pred_mu = mu

        # Build obs in fixed order: [μ, σ_eff, sharpe, (ood?), (corr?), (autocorr?), regime, portfolio]
        head = [mu, sig_eff, sharpe]
        if self.use_ood_attenuation:
            head.append(ood_score)
        if self.use_ewma_corr:
            head.append(recent_corr)
        if self.use_autocorr_feature:
            head.append(autocorr)

        return np.concatenate([
            np.array(head, dtype=np.float32),
            regime.astype(np.float32),
            portfolio,
        ])

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.base_env.reset(seed=seed, options=options)
        # Reset diagnostics + EWMA state per episode
        self._last_vol_scaler = 1.0
        self._last_ood_score  = 0.0
        self._last_recent_corr = 0.0
        self._reset_ewma_state()
        self._pending_pred_mu = None
        return self._compact_obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()

        # ── v3.1 vol-targeting attenuation ──────────────────────────────────
        if self.target_vol is not None:
            rh = self.base_env.return_history
            if len(rh) >= self.vol_window:
                realized = float(np.std(rh[-self.vol_window:]) * np.sqrt(252))
                realized = max(realized, 0.05)
                scaler = float(np.clip(
                    self.target_vol / realized,
                    self.vol_scaler_min, self.vol_scaler_max,
                ))
                action *= scaler
                self._last_vol_scaler = scaler
            else:
                self._last_vol_scaler = 1.0

        # ── v3.2 OOD attenuation: shrink action when forecasters disagree ───
        if self.use_ood_attenuation and self._last_ood_score is not None:
            ood = self._last_ood_score
            ood_mult = 1.0 - self.ood_attenuation_alpha * ood
            action *= ood_mult

        # Capture index BEFORE base step (current_step still points at "now")
        idx_now = self.base_env.start_idx + self.base_env.lookback + self.base_env.current_step

        _, reward, terminated, truncated, info = self.base_env.step(action)

        # ── v3.2 update EWMA correlation with (predicted_μ_now, realised_ret_now) ──
        if self.use_ewma_corr and self._pending_pred_mu is not None:
            if 0 <= idx_now < len(self.base_env.return_arr):
                r_realised = float(self.base_env.return_arr[idx_now])
                self._update_ewma(self._pending_pred_mu, r_realised)

        return self._compact_obs(), reward, terminated, truncated, info

    def _info(self) -> dict:
        return self.base_env._get_info()

    def render(self): return self.base_env.render()
    def close(self):  return self.base_env.close()

    # Pass-through helpers for evaluation
    def get_episode_metrics(self):
        return self.base_env.get_episode_metrics()

    @property
    def df(self):              return self.base_env.df
    @property
    def features(self):        return self.base_env.features
    @property
    def start_idx(self):       return self.base_env.start_idx
    @property
    def lookback(self):        return self.base_env.lookback
    @property
    def current_step(self):    return self.base_env.current_step
    @property
    def return_history(self):  return self.base_env.return_history


# ─────────────────────────────────────────────────────────────────────────────
# Constraint-aware action head
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintAwareActionNet(nn.Module):
    """
    Compact action head with fractional-Kelly constraint (Thorp 1969 form).

    Architecture
    ------------
      shared trunk   :  state → h
      direction head :  Tanh — outputs ∈ [-1, +1]
      magnitude head :  Sigmoid — outputs ∈ [0, 1]

    Final action  =  direction × magnitude × kelly(state)

      kelly = clamp( kelly_fraction × |μ / σ²|, 0, 1.0 )

    where `kelly_fraction` is the Thorp-style fractional-Kelly multiplier
    (1.0 = full Kelly, 0.5 = half-Kelly, etc.) and the final clamp at 1.0
    enforces 100%-of-bankroll as the absolute leverage ceiling.

    Why fraction-then-clip, not flat cap?
    -------------------------------------
    The previous formulation ``kelly = clamp(|μ/σ²|, 0, cap)`` treated
    ``cap`` as a hard ceiling.  When ``cap = 0.5`` and the Kelly
    recommendation is 1.6 (typical for confident forecasts), the action was
    crushed to 0.5 — eliminating the upside that would offset transaction
    costs.  PPO then learned a degenerate "don't trade" policy.

    The new form scales the Kelly recommendation by ``kelly_fraction`` and
    only clips at 100% leverage.  At ``kelly_fraction=0.5`` and Kelly=1.6,
    we get kelly=0.8 (still ample headroom for upside), while weak
    forecasts (Kelly=0.4) give kelly=0.2 (correctly attenuated).

    Reference
    ---------
    Kelly (1956); Thorp (1969) "Optimal Gambling Systems for Favorable Games".
    """

    def __init__(
        self,
        state_dim: int,
        hidden: int                                  = 64,
        kelly_fraction: float                        = 0.5,
        kelly_per_regime: Optional[np.ndarray]       = None,
        regime_obs_offset: int                       = 3,
        n_regimes: int                               = 4,
        # v3.3
        detach_kelly: bool                           = True,
    ) -> None:
        super().__init__()
        self.kelly_fraction    = kelly_fraction
        self.regime_obs_offset = int(regime_obs_offset)
        self.n_regimes         = int(n_regimes)
        self.detach_kelly      = bool(detach_kelly)

        if kelly_per_regime is not None:
            kpr = np.asarray(kelly_per_regime, dtype=np.float32)
            if kpr.shape[0] != n_regimes:
                raise ValueError(
                    f"kelly_per_regime length {kpr.shape[0]} != n_regimes {n_regimes}"
                )
            # Register as buffer so .to(device) moves it correctly.
            self.register_buffer("kelly_per_regime", th.from_numpy(kpr))
            self._use_regime_kelly = True
        else:
            self.register_buffer("kelly_per_regime", th.zeros(n_regimes))
            self._use_regime_kelly = False

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.dir_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())
        self.mag_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        # Diagnostic state
        self._last_kelly: Optional[np.ndarray]      = None
        self._last_direction: Optional[np.ndarray]  = None
        self._last_magnitude: Optional[np.ndarray]  = None
        self._last_kelly_frac: Optional[np.ndarray] = None

    def forward(self, state: th.Tensor) -> th.Tensor:
        h = self.trunk(state)
        direction = self.dir_head(h)            # (B, 1) ∈ [-1, +1]
        magnitude = self.mag_head(h)            # (B, 1) ∈ [0, 1]

        mu  = state[:, 0:1]
        sig = state[:, 1:2].clamp(min=1e-6)

        # v3.1/v3.3 — regime-weighted Kelly fraction (with per-regime hard caps)
        if self._use_regime_kelly:
            r = state[:, self.regime_obs_offset : self.regime_obs_offset + self.n_regimes]
            r_norm = r / (r.sum(dim=-1, keepdim=True) + 1e-8)
            kelly_frac = (r_norm * self.kelly_per_regime.unsqueeze(0)).sum(dim=-1, keepdim=True)
        else:
            kelly_frac = th.full_like(mu, self.kelly_fraction)

        kelly_optimal = (mu / (sig**2)).abs()
        kelly = (kelly_frac * kelly_optimal).clamp(0.0, 1.0)

        # v3.3 — detach Kelly to prevent perverse gradient dynamics
        # (rationale: PPO's gradients shouldn't be amplified on high-Kelly days)
        if self.detach_kelly:
            kelly_used = kelly.detach()
        else:
            kelly_used = kelly

        action = direction * magnitude * kelly_used  # (B, 1)

        # Diagnostics
        self._last_kelly      = kelly.detach().cpu().numpy()
        self._last_direction  = direction.detach().cpu().numpy()
        self._last_magnitude  = magnitude.detach().cpu().numpy()
        self._last_kelly_frac = kelly_frac.detach().cpu().numpy()

        return action.clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Compact actor-critic policy for SB3 PPO
# ─────────────────────────────────────────────────────────────────────────────

class CompactConstraintPolicy(ActorCriticPolicy):
    """
    Drop-in MlpPolicy replacement for SB3 PPO with:
      - Tiny MLP value head
      - ConstraintAwareActionNet (Kelly-bounded continuous action)
      - log_std floor at -1.0 (carried over from v2)

    Constructor kwargs (beyond standard SB3):
      hidden          int   64
      kelly_fraction  float 0.5   fractional-Kelly multiplier (Thorp 1969 form);
                                  1.0 = full Kelly, 0.5 = half-Kelly
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        hidden: int                                = 64,
        kelly_fraction: float                      = 0.5,
        kelly_per_regime: Optional[np.ndarray]     = None,
        n_regimes: int                             = 4,
        regime_obs_offset: int                     = 3,
        detach_kelly: bool                         = True,   # v3.3
        **kwargs,
    ) -> None:
        self._hidden              = hidden
        self._kelly_fraction      = kelly_fraction
        self._kelly_per_regime    = kelly_per_regime
        self._n_regimes           = n_regimes
        self._regime_obs_offset   = regime_obs_offset
        self._detach_kelly        = detach_kelly
        kwargs["net_arch"] = []
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        action_dim = int(np.prod(self.action_space.shape))

        # log_std with floor at -1.0 (same trick as v2 — prevents entropy collapse)
        self.log_std = nn.Parameter(
            th.ones(action_dim) * self.log_std_init, requires_grad=True
        )

        state_dim = int(self.observation_space.shape[0])
        self.action_net = ConstraintAwareActionNet(
            state_dim        = state_dim,
            hidden           = self._hidden,
            kelly_fraction   = self._kelly_fraction,
            kelly_per_regime = self._kelly_per_regime,
            n_regimes        = self._n_regimes,
            regime_obs_offset= self._regime_obs_offset,
            detach_kelly     = self._detach_kelly,
        )

        # Compact value head
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, self._hidden), nn.ReLU(),
            nn.Linear(self._hidden, self._hidden), nn.ReLU(),
            nn.Linear(self._hidden, 1),
        )

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        mean_actions = self.action_net(latent_pi)
        # v3.3: tighter floor (−1.5) so exploration noise doesn't dominate the
        # signal.  σ_exploration = exp(-1.5) ≈ 0.22 (was 0.37 at floor=-1.0)
        log_std_floored = th.clamp(self.log_std, min=-1.5)
        return self.action_dist.proba_distribution(mean_actions, log_std_floored)

    @property
    def last_kelly(self) -> Optional[float]:
        k = getattr(self.action_net, "_last_kelly", None)
        return float(k.mean()) if k is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_compact_ppo(
    env,
    total_timesteps: int   = 200_000,
    learning_rate: float   = 3e-4,
    n_steps: int           = 1024,
    batch_size: int        = 64,
    n_epochs: int          = 5,
    gamma: float           = 0.97,
    gae_lambda: float      = 0.95,
    clip_range: float      = 0.2,
    vf_coef: float         = 0.5,
    max_grad_norm: float   = 0.5,
    ent_coef: float        = 0.01,
    hidden: int                              = 64,
    kelly_fraction: float                    = 0.5,
    kelly_per_regime: Optional[np.ndarray]   = None,
    n_regimes: int                           = 4,
    regime_obs_offset: int                   = 3,
    detach_kelly: bool                       = True,   # v3.3
    device: str                              = "auto",
    seed: int                                = 42,
    verbose: int                             = 0,
) -> PPO:
    """
    Build a PPO model with the CompactConstraintPolicy on a compact state
    space.  Designed for the HybridForecastEnv.

    `kelly_fraction` is the Thorp-style fractional-Kelly multiplier:
      1.0  full Kelly    — aggressive, can over-leverage on noisy forecasts
      0.5  half-Kelly    — practitioner default, balanced
      0.25 quarter-Kelly — defensive

    Default hyperparameters reflect the much smaller policy (~5-10k params)
    and the better signal-to-noise ratio:
      - higher LR than v2 (3e-4 vs 3e-6) — small net, less overfitting risk
      - vanilla PPO defaults (no MoE, no router entropy bonus)
    """
    policy_kwargs: Dict[str, Any] = dict(
        hidden            = hidden,
        kelly_fraction    = kelly_fraction,
        kelly_per_regime  = kelly_per_regime,
        n_regimes         = n_regimes,
        regime_obs_offset = regime_obs_offset,
        detach_kelly      = detach_kelly,
    )
    return PPO(
        policy        = CompactConstraintPolicy,
        env           = env,
        learning_rate = learning_rate,
        n_steps       = n_steps,
        batch_size    = batch_size,
        n_epochs      = n_epochs,
        gamma         = gamma,
        gae_lambda    = gae_lambda,
        clip_range    = clip_range,
        ent_coef      = ent_coef,
        vf_coef       = vf_coef,
        max_grad_norm = max_grad_norm,
        policy_kwargs = policy_kwargs,
        device        = device,
        seed          = seed,
        verbose       = verbose,
    )
