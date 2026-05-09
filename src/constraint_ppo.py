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
    ) -> None:
        super().__init__()
        self.base_env = base_env
        self.forecast_mus    = forecast_mus.astype(np.float32)
        self.forecast_sigmas = forecast_sigmas.astype(np.float32)
        self.regime_probs    = regime_probs.astype(np.float32)
        self.ensemble        = ensemble

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
        self._obs_dim = 3 + n_regimes + 5  # μ, σ, sharpe, R regimes, 5 portfolio
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = base_env.action_space

    # ── Compact observation builder ───────────────────────────────────────────

    def _compact_obs(self) -> np.ndarray:
        # Index into the precomputed forecasts at the current time
        idx = self.base_env.start_idx + self.base_env.lookback + self.base_env.current_step
        idx = min(idx, len(self._ensemble_pred) - 1)

        mu  = float(self._ensemble_pred[idx, 0])
        sig = float(max(self._ensemble_pred[idx, 1], 1e-6))
        sharpe = float(np.clip(mu / sig, -5.0, 5.0))
        regime = self.regime_probs[idx]                       # (R,)

        # Portfolio state (mirror the last 5 entries of base env's obs)
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

        obs = np.concatenate([
            np.array([mu, sig, sharpe], dtype=np.float32),
            regime.astype(np.float32),
            portfolio,
        ])
        return obs

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.base_env.reset(seed=seed, options=options)
        return self._compact_obs(), self._info()

    def step(self, action):
        _, reward, terminated, truncated, info = self.base_env.step(action)
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
        hidden: int             = 64,
        kelly_fraction: float   = 0.5,    # 1.0 = full Kelly, 0.5 = half-Kelly
    ) -> None:
        super().__init__()
        self.kelly_fraction = kelly_fraction
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.dir_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())
        self.mag_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        # Diagnostic state
        self._last_kelly: Optional[np.ndarray]    = None
        self._last_direction: Optional[np.ndarray] = None
        self._last_magnitude: Optional[np.ndarray] = None

    def forward(self, state: th.Tensor) -> th.Tensor:
        h = self.trunk(state)
        direction = self.dir_head(h)            # (B, 1) ∈ [-1, +1]
        magnitude = self.mag_head(h)            # (B, 1) ∈ [0, 1]

        mu  = state[:, 0:1]
        sig = state[:, 1:2].clamp(min=1e-6)
        kelly_optimal = (mu / (sig**2)).abs()                     # (B, 1)
        kelly = (self.kelly_fraction * kelly_optimal).clamp(0.0, 1.0)

        action = direction * magnitude * kelly  # (B, 1)

        # Diagnostics
        self._last_kelly     = kelly.detach().cpu().numpy()
        self._last_direction = direction.detach().cpu().numpy()
        self._last_magnitude = magnitude.detach().cpu().numpy()

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
        hidden: int            = 64,
        kelly_fraction: float  = 0.5,
        **kwargs,
    ) -> None:
        self._hidden         = hidden
        self._kelly_fraction = kelly_fraction
        kwargs["net_arch"] = []  # we build our own
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
            state_dim=state_dim,
            hidden=self._hidden,
            kelly_fraction=self._kelly_fraction,
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
        log_std_floored = th.clamp(self.log_std, min=-1.0)
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
    hidden: int            = 64,
    kelly_fraction: float  = 0.5,
    device: str            = "auto",
    seed: int              = 42,
    verbose: int           = 0,
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
        hidden         = hidden,
        kelly_fraction = kelly_fraction,
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
