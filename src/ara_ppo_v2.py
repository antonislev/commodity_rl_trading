"""
ARA-PPO v2 — Mixture-of-Experts PPO for WTI Crude Oil Trading
==============================================================
Fixes implemented
-----------------
  Phase 2.1  MoE policy with 1D-CNN temporal encoder
  Phase 2.2  Regime-aware critic (obs + regime → value)
  Phase 4.1  Asymmetric reward environment wrapper
  Phase 4.2  Regime-weighted rollout buffer (over-samples crisis steps)

Stability fixes (v2.1) — applied after splits 1-10 showed 52% collapse rate
-----------------------------------------------------------------------------
  Fix C1  log_std hard floor at -1.0 via _get_action_dist_from_latent override
          → entropy never falls below H_min ≈ 0.42 nats
  Fix C2  n_steps 256 → 1024: longer rollouts for better GAE estimates
  Fix C3  n_epochs 12 → 5: per-sample IS-ratio drift 12× → 5×

Architecture improvements (v2.2) — choppy-market failure mode fix
-----------------------------------------------------------------
  Fix D1  TemporalAttention replaces AdaptiveAvgPool in CNN encoder
          → model can weight-differ recent vs older time positions
          → receptive field extends to full 60-day window via attention
          cost: +16.8 k params (MultiheadAttention 64-dim, 4 heads)

  Fix D2  HybridMoEActionNet replaces MoEActionNet
          → discrete gate (long / short / flat) + continuous size head
          → flat is a first-class learnable action, not "output ≈ 0.0"
          → gate trained with Gumbel-softmax (straight-through estimator)
          → gate entropy bonus encourages using the flat option

  See environment.py for direction_change_penalty (reward-side complement).
  See callbacks_v2.py for RouterMonitorCallback gate-prob logging.

Entropy scheduling and collapse monitoring live in callbacks_v2.py.

Architecture summary (v2.2)
---------------------------
  CNN temporal encoder   (47 features × 60 days)                        30 208 params
    Conv1d stack: 47→32→64→64 (stride 2)                                30 208 params
  TemporalAttention      8 time steps × 64 channels, 4 heads            16 768 params
  Portfolio encoder       5 dims → 8 dims                                    48 params
  Projection              72 → 64                                          4 672 params
  Features extractor out  [64-dim embedding | 4 regime signals] = 68 dim
  3 × Expert feature enc  64 → 32                                          6 240 params
  Router                  4 → 16 → 3  + scalar τ                            132 params
  Gate head               32 → 3  (long / short / flat)                      99 params
  Size head               32 → 1  + Sigmoid                                  33 params
  Regime critic           68 → 64 → 32 → 1                                6 529 params
  log_std                                                                       1 param
  ──────────────────────────────────────────────────────────────────────────────────────
  Total                                                                    64 730 params
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device

import gymnasium as gym

# ─────────────────────────────────────────────────────────────────────────────
# Constants (match feature_config.json and environment.py)
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES   = 47   # engineered features per timestep
N_TIMESTEPS  = 60   # lookback window
N_PORTFOLIO  = 5    # portfolio state dims appended to obs
N_REGIME     = 4    # last N_REGIME features in each timestep are regime signals
CNN_OUT      = 64   # CNN output embedding size
N_EXPERTS    = 3    # number of specialist sub-policies

# Index of the first regime feature in the last timestep of the flattened obs:
#   regime_start = (N_TIMESTEPS-1)*N_FEATURES + (N_FEATURES-N_REGIME)
#                = 59*47 + 43 = 2816
REGIME_IDX_START = (N_TIMESTEPS - 1) * N_FEATURES + (N_FEATURES - N_REGIME)
REGIME_IDX_END   = REGIME_IDX_START + N_REGIME  # 2820

# Index used to identify crisis steps in the rollout buffer (regime_vol,
# the first regime feature in the last timestep).
REGIME_VOL_IDX = REGIME_IDX_START  # obs[:, 2816]


# ─────────────────────────────────────────────────────────────────────────────
# v2.2 Fix D1 — Temporal Self-Attention (replaces AdaptiveAvgPool in CNN)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Single-layer multi-head self-attention over the compressed temporal
    dimension that comes out of the 1D-CNN stack.

    Why replace AdaptiveAvgPool?
    ----------------------------
    AdaptiveAvgPool treats all time positions equally — it is a simple mean
    of the last 8 compressed time steps.  In trending markets this is fine.
    In choppy / range-bound markets the most informative signal is often the
    *most recent* period (is momentum decelerating?) or a specific earlier
    anchor (where was the last resistance level?).  Attention learns to assign
    non-uniform importance weights across the 8 positions.

    The CNN already compresses 60 → 8 timesteps via stride-2 convolutions.
    Attention over 8 short sequences is cheap: O(8²) = 64 operations, vs
    O(60²) = 3 600 if applied to the raw window.

    Parameters: 16 768  (MultiheadAttention 64-dim + 4 heads + LayerNorm)

    Input:  (batch, channels=64, time=8)   ← from CNN last Conv1d layer
    Output: (batch, channels=64)           ← attended temporal mean
    """

    def __init__(self, channels: int = CNN_OUT, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: (B, C, T) from Conv1d — reshape for MultiheadAttention
        x = x.permute(0, 2, 1)               # (B, T, C)
        attn_out, _ = self.attn(x, x, x)     # self-attention over T positions
        x = self.norm(x + attn_out)           # residual + layer norm
        return x.mean(dim=1)                  # (B, C) temporal mean pool


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.1 — 1D-CNN Temporal Encoder (Features Extractor)
# ─────────────────────────────────────────────────────────────────────────────

class CNNTemporalEncoder(BaseFeaturesExtractor):
    """
    Compress a (60 × 47) market observation window into a 64-dim embedding,
    then concatenate the 4 most-recent regime signals so the downstream MoE
    router can always see them without an extra extraction step.

    Output dimension: CNN_OUT + N_REGIME = 68
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        cnn_out: int = CNN_OUT,
        n_regime: int = N_REGIME,
        n_features: int = N_FEATURES,
        n_timesteps: int = N_TIMESTEPS,
        n_portfolio: int = N_PORTFOLIO,
    ) -> None:
        features_dim = cnn_out + n_regime  # 68
        super().__init__(observation_space, features_dim)

        self.cnn_out     = cnn_out
        self.n_regime    = n_regime
        self.n_features  = n_features
        self.n_timesteps = n_timesteps
        self.n_portfolio = n_portfolio

        # Regime signal slice indices in the raw obs vector
        self._regime_start = (n_timesteps - 1) * n_features + (n_features - n_regime)
        self._regime_end   = self._regime_start + n_regime

        # ── 1D-CNN over time dimension ────────────────────────────────────────
        # Input:  (batch, n_features=47, n_timesteps=60)
        # Output: (batch, 64, 8) — 8 compressed time steps
        #
        # Receptive-field walkthrough (stride=2 each layer):
        #   Conv1: (47, 60) → (32, 30)   7 552 params
        #   Conv2: (32, 30) → (64, 15)  10 304 params
        #   Conv3: (64, 15) → (64,  8)  12 352 params
        #   v2.2: AdaptiveAvgPool replaced by TemporalAttention (see below)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, cnn_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Note: no AdaptiveAvgPool here — TemporalAttention does the pooling
        )

        # ── Temporal self-attention over the 8 compressed time steps ─────────
        self.temporal_attn = TemporalAttention(cnn_out, n_heads=4)

        # ── Portfolio state encoder ───────────────────────────────────────────
        # [position, drawdown, log_pnl, time_progress, cost_ratio]
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(n_portfolio, 8),
            nn.ReLU(),
        )

        # ── Merge CNN + portfolio → cnn_out ──────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(cnn_out + 8, cnn_out),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch = observations.shape[0]
        market_end = self.n_timesteps * self.n_features

        # ── Split raw obs ─────────────────────────────────────────────────────
        market_flat   = observations[:, :market_end]          # (B, 2820)
        portfolio_obs = observations[:, market_end:]           # (B, 5)

        # Extract regime signals from the last timestep (no lookahead — these
        # are the features at t-0, already in the obs).
        regime_signals = observations[:, self._regime_start:self._regime_end]  # (B, 4)

        # ── CNN encoding ──────────────────────────────────────────────────────
        # Reshape to (batch, n_features, n_timesteps) for Conv1d
        x = market_flat.view(batch, self.n_timesteps, self.n_features)
        x = x.permute(0, 2, 1).contiguous()       # (B, 47, 60)
        cnn_feat = self.cnn(x)                     # (B, 64, 8)
        cnn_out  = self.temporal_attn(cnn_feat)    # (B, 64) — attended mean pool

        # ── Portfolio encoding ────────────────────────────────────────────────
        p = self.portfolio_encoder(portfolio_obs)  # (B, 8)

        # ── Merge & project ───────────────────────────────────────────────────
        combined  = th.cat([cnn_out, p], dim=-1)       # (B, 72)
        embedding = self.projection(combined)           # (B, 64)

        # ── Concatenate regime signals (for router) ───────────────────────────
        return th.cat([embedding, regime_signals], dim=-1)  # (B, 68)


# ─────────────────────────────────────────────────────────────────────────────
# v2.2 Fix D2 — Hybrid MoE Action Network (gate + size)
# ─────────────────────────────────────────────────────────────────────────────

class HybridMoEActionNet(nn.Module):
    """
    Three specialist sub-policies blended by a regime router, with a hybrid
    gate + size output head that gives the agent a clean "flat" action.

    Why a hybrid head?
    ------------------
    The previous MoEActionNet output a single weighted-mean scalar ∈ [-1, +1].
    To stay flat the policy had to output *exactly* 0.0 from a Gaussian
    distribution — that is a measure-zero event it can never reliably learn.
    In choppy markets (splits 13-14, 17-20, 30-32) the agent therefore
    traded noise, accumulating small losses and transaction costs.

    The hybrid head decomposes the decision into two independent signals:
      • Gate   : categorical (long / short / flat) — the direction decision
      • Size   : continuous magnitude in [0, 1]    — how much to trade

    Final action = gate_direction × size  ∈  {-1, 0, +1} × [0, 1]

    Training: Gumbel-softmax (hard=True) keeps the gate differentiable while
    producing sharp one-hot samples.  Standard gradients flow through both the
    direction and size heads via the straight-through estimator.

    Eval: hard argmax — deterministic, fast, interpretable.

    The flat gate receives a gate-entropy bonus in evaluate_actions() that
    explicitly rewards using the flat option, analogous to the router entropy
    bonus that prevents expert collapse.

    Architecture (params)
    ---------------------
      3 × Expert feature encoders: Linear(64→32)→ReLU          6 240
      Regime router:                Linear(4→16)→ReLU→Linear(16→3) + τ   132
      Gate head:                    Linear(32→3)                   99
      Size head:                    Linear(32→1)→Sigmoid            33
      ─────────────────────────────────────────────────────────────────
      Total                                                      6 504
    """

    LONG  = 0
    SHORT = 1
    FLAT  = 2

    def __init__(
        self,
        cnn_out: int       = CNN_OUT,
        n_regime: int      = N_REGIME,
        n_experts: int     = N_EXPERTS,
        expert_hidden: int = 32,
        action_dim: int    = 1,         # kept for API compat; only 1 supported
        gumbel_tau: float  = 1.0,
    ) -> None:
        super().__init__()
        self.cnn_out      = cnn_out
        self.n_regime     = n_regime
        self.n_experts    = n_experts
        self.gumbel_tau   = gumbel_tau

        # Expert feature encoders (no final action logit — they produce features)
        # Each: Linear(64→32)→ReLU  — experts specialise their 32-dim features
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(cnn_out, expert_hidden), nn.ReLU())
            for _ in range(n_experts)
        ])

        # Regime router (reads ONLY regime signals — keeps routing decision
        # cleanly separated from price-action embedding)
        self.router = nn.Sequential(
            nn.Linear(n_regime, 16),
            nn.ReLU(),
            nn.Linear(16, n_experts),
        )
        # Learnable softmax temperature (init 1.0, clamped ≥ 0.1)
        self.tau = nn.Parameter(th.ones(1))

        # Gate head: blended expert features → P(long, short, flat)
        self.gate = nn.Linear(expert_hidden, 3)

        # Size head: blended expert features → position magnitude in [0, 1]
        self.size_head = nn.Sequential(
            nn.Linear(expert_hidden, 1),
            nn.Sigmoid(),
        )

        # ── Storage for monitoring and entropy regularisation ─────────────────
        self._router_weights_grad: Optional[th.Tensor] = None  # graph attached
        self._last_weights: Optional[np.ndarray]       = None  # detached, numpy
        self._gate_probs_grad: Optional[th.Tensor]     = None  # graph attached
        self._last_gate_probs: Optional[np.ndarray]    = None  # detached, numpy

    def forward(self, latent: th.Tensor) -> th.Tensor:
        """
        latent: (B, cnn_out + n_regime) from CNNTemporalEncoder
        Returns: (B, 1) action in [-1, +1]
          = 0.0 exactly when gate = FLAT (eval) or gate_sample = flat (train)
        """
        embedding = latent[:, :self.cnn_out]    # (B, 64)
        regime    = latent[:, self.cnn_out:]    # (B,  4)

        # ── Expert feature vectors ────────────────────────────────────────────
        expert_feats = th.stack(
            [e(embedding) for e in self.experts], dim=1
        )  # (B, n_experts, expert_hidden)

        # ── Router weights (regime-conditioned) ──────────────────────────────
        router_logits = self.router(regime)                                    # (B, K)
        weights       = F.softmax(router_logits / self.tau.clamp(min=0.1), dim=-1)

        self._router_weights_grad = weights
        self._last_weights        = weights.detach().cpu().numpy()

        # ── Blended expert features ───────────────────────────────────────────
        blended = (expert_feats * weights.unsqueeze(-1)).sum(dim=1)  # (B, 32)

        # ── Gate: direction decision ──────────────────────────────────────────
        gate_logits = self.gate(blended)                 # (B, 3)
        gate_probs  = F.softmax(gate_logits, dim=-1)     # (B, 3)

        self._gate_probs_grad = gate_probs               # WITH gradient
        self._last_gate_probs = gate_probs.detach().cpu().numpy()

        # ── Size: magnitude ───────────────────────────────────────────────────
        size = self.size_head(blended)                   # (B, 1) ∈ [0, 1]

        # ── Differentiable gate decision ──────────────────────────────────────
        # Training: Gumbel-softmax hard=True (straight-through estimator)
        #   Forward pass: sharp one-hot sample (stops gradient w.r.t. the sample)
        #   Backward pass: gradients flow through the soft probabilities
        #
        # Eval: SOFT gate probabilities — NOT hard argmax.
        #
        # WHY soft eval, not argmax?
        # Hard argmax turns a learned soft flat-preference into 100% flat.
        # Example: if gate_probs = [P(long)=0.35, P(short)=0.25, P(flat)=0.40]
        #   argmax → FLAT → direction=0 → action=0 for EVERY observation.
        #   This is the root cause of trade_count=0 / Sharpe=0 at eval time.
        # Soft eval: direction = P(long) - P(short) = 0.35 - 0.25 = 0.10 (long)
        #   Action magnitude is proportional to directional confidence,
        #   never collapses to a constant zero.
        if self.training:
            gate_sample = F.gumbel_softmax(
                gate_logits, tau=self.gumbel_tau, hard=True
            )  # (B, 3) one-hot
            direction = (
                gate_sample[:, self.LONG] - gate_sample[:, self.SHORT]
            ).unsqueeze(-1)                              # (B, 1) ∈ {-1, 0, +1}
        else:
            # Soft eval: continuous direction signal in (-1, +1)
            gate_probs_eval = F.softmax(gate_logits, dim=-1)  # (B, 3)
            direction = (
                gate_probs_eval[:, self.LONG] - gate_probs_eval[:, self.SHORT]
            ).unsqueeze(-1)                              # (B, 1) ∈ (-1, +1)

        return direction * size                          # (B, 1) ∈ [-1, +1]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.2 — Regime-Aware Critic
# ─────────────────────────────────────────────────────────────────────────────

class RegimeValueNet(nn.Module):
    """
    Single-stream critic that takes the full 68-dim features extractor output
    (embedding + regime signals) and produces a scalar value estimate.

    Architecture: Linear(68,64)→ReLU→Linear(64,32)→ReLU→Linear(32,1)
    Parameters: 6 529
    """

    def __init__(self, cnn_out: int = CNN_OUT, n_regime: int = N_REGIME) -> None:
        super().__init__()
        in_dim = cnn_out + n_regime  # 68
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent: th.Tensor) -> th.Tensor:
        return self.net(latent)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.1+2.2 — MoE Actor-Critic Policy
# ─────────────────────────────────────────────────────────────────────────────

class MoEActorCriticPolicy(ActorCriticPolicy):
    """
    Drop-in replacement for SB3's default MlpPolicy.

    Key differences from default SB3 ActorCriticPolicy:
      • features_extractor = CNNTemporalEncoder (1D-CNN, outputs 68 dims)
      • action_net         = MoEActionNet       (3 experts + regime router)
      • value_net          = RegimeValueNet     (obs+regime → scalar)
      • evaluate_actions() folds router-entropy into action-entropy for
        load-balance regularisation (no PPO subclassing required)

    Constructor kwargs (beyond standard SB3):
      n_experts       int   3
      expert_hidden   int   32
      router_ent_coef float 0.1   scale on router entropy bonus (expert load-balance)
      gate_ent_coef   float 0.05  scale on gate entropy bonus (encourages flat usage)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # Architecture
        n_experts: int       = N_EXPERTS,
        expert_hidden: int   = 32,
        cnn_out: int         = CNN_OUT,
        n_regime: int        = N_REGIME,
        n_features: int      = N_FEATURES,
        n_timesteps: int     = N_TIMESTEPS,
        n_portfolio: int     = N_PORTFOLIO,
        # Load-balance regularisation
        router_ent_coef: float = 0.1,
        # Gate entropy bonus — encourages using the flat option
        gate_ent_coef: float   = 0.05,
        **kwargs: Any,
    ) -> None:
        # Store before super().__init__ because _build() is called inside it
        self._n_experts       = n_experts
        self._expert_hidden   = expert_hidden
        self._cnn_out         = cnn_out
        self._n_regime        = n_regime
        self._router_ent_coef = router_ent_coef
        self._gate_ent_coef   = gate_ent_coef

        # Force pass-through MLP extractor (net_arch=[]) and our CNN extractor
        kwargs["net_arch"] = []
        kwargs["features_extractor_class"]  = CNNTemporalEncoder
        kwargs["features_extractor_kwargs"] = dict(
            cnn_out      = cnn_out,
            n_regime     = n_regime,
            n_features   = n_features,
            n_timesteps  = n_timesteps,
            n_portfolio  = n_portfolio,
        )

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ── Override _build to replace default action_net and value_net ───────────

    def _build(self, lr_schedule: Schedule) -> None:
        # 1. Build pass-through mlp_extractor (net_arch=[])
        self._build_mlp_extractor()

        # 2. Action dimension
        action_dim = int(np.prod(self.action_space.shape))

        # 3. log_std parameter (required by DiagGaussianDistribution in SB3)
        self.log_std = nn.Parameter(
            th.ones(action_dim) * self.log_std_init, requires_grad=True
        )

        # 4. Hybrid MoE actor head (gate + size, v2.2)
        self.action_net = HybridMoEActionNet(
            cnn_out       = self._cnn_out,
            n_regime      = self._n_regime,
            n_experts     = self._n_experts,
            expert_hidden = self._expert_hidden,
            action_dim    = action_dim,
        )

        # 5. Regime-aware critic
        self.value_net = RegimeValueNet(
            cnn_out  = self._cnn_out,
            n_regime = self._n_regime,
        )

        # 6. Optimizer over ALL parameters (features_extractor + action_net +
        #    value_net + log_std)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ── Override evaluate_actions to add router-entropy load-balance bonus ────

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Standard SB3 signature: returns (values, log_prob, entropy).

        Additionally adds a router-entropy bonus to `entropy` so that the PPO
        entropy loss automatically encourages balanced expert usage.

        Router entropy bonus:
          H_router = -Σ w_i log(w_i)   (per sample, shape B)
          effective entropy = H_action + router_ent_coef * H_router
        """
        features             = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob     = distribution.log_prob(actions)
        entropy      = distribution.entropy()  # (B,)
        values       = self.value_net(latent_vf)

        # ── Router load-balance entropy bonus ────────────────────────────────
        # Encourages balanced expert usage.  _router_weights_grad was set
        # inside action_net.forward() (called from _get_action_dist_from_latent)
        # and is still in the computation graph.
        if (
            self._router_ent_coef > 0.0
            and hasattr(self.action_net, "_router_weights_grad")
            and self.action_net._router_weights_grad is not None
        ):
            w = self.action_net._router_weights_grad          # (B, n_experts)
            router_entropy = -(w * th.log(w + 1e-8)).sum(-1)  # (B,)
            entropy = entropy + self._router_ent_coef * router_entropy

        # ── Gate entropy bonus ────────────────────────────────────────────────
        # Encourages the gate to actually use the flat option.  Without this
        # the gate can collapse to always-long or always-short in trending
        # periods and never recover the flat behaviour for choppy periods.
        if (
            self._gate_ent_coef > 0.0
            and hasattr(self.action_net, "_gate_probs_grad")
            and self.action_net._gate_probs_grad is not None
        ):
            g = self.action_net._gate_probs_grad              # (B, 3)
            gate_entropy = -(g * th.log(g + 1e-8)).sum(-1)    # (B,)
            entropy = entropy + self._gate_ent_coef * gate_entropy

        return values, log_prob, entropy

    # ── Override _get_action_dist_from_latent to enforce log_std floor ────────

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        """
        Override SB3 default to clamp log_std from below, preventing the
        policy from collapsing to a near-deterministic delta function.

        At log_std = -1.0:
          H = 0.5 * (1 + log(2π)) + log_std ≈ 0.42 nats
        This keeps enough exploration to recover from bad patches while still
        allowing the policy to commit to high-confidence actions.

        Without this floor: 48 gradient updates/rollout (4 batches × 12 epochs)
        drive log_std negative without bound, causing entropy → 0 in ~10 rollouts.
        """
        mean_actions = self.action_net(latent_pi)
        log_std_floored = th.clamp(self.log_std, min=-1.0)
        return self.action_dist.proba_distribution(mean_actions, log_std_floored)

    # ── Helpers: expose router and gate state for logging ────────────────────

    @property
    def last_router_weights(self) -> Optional[np.ndarray]:
        """Last batch's router weights as numpy array (n_experts,)."""
        if self.action_net._last_weights is not None:
            return self.action_net._last_weights.mean(axis=0)
        return None

    @property
    def last_gate_probs(self) -> Optional[np.ndarray]:
        """Last batch's gate probabilities as numpy array (3,) = [P(long), P(short), P(flat)]."""
        if (hasattr(self.action_net, '_last_gate_probs')
                and self.action_net._last_gate_probs is not None):
            return self.action_net._last_gate_probs.mean(axis=0)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4.2 — Regime-Weighted Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class RegimeWeightedRolloutBuffer(RolloutBuffer):
    """
    Subclass of SB3's RolloutBuffer that biases PPO minibatches toward
    high-volatility / crisis regime steps.

    Design: PPO-safe stratified ordering
    -------------------------------------
    The original implementation used ``replace=True`` sampling, which caused
    a single crisis step to appear up to 36× per rollout (3× oversample ×
    12 epochs), exploding the importance-sampling ratio and collapsing the
    policy.

    The corrected approach keeps each step to AT MOST ONE appearance per
    epoch-pass (PPO-safe) while still biasing the ORDER of the permutation
    so crisis steps are drawn into earlier minibatches.  Concretely:

    1. Assign sampling weights (crisis = ``crisis_oversample``, others = 1.0).
    2. Normalise to a probability vector p.
    3. Draw a weighted permutation of size n_total WITHOUT replacement using
       Gumbel-max trick: sort by (log p_i + Gumbel noise).  This produces a
       random ordering where high-weight indices tend to appear first.
    4. Slice the permutation into minibatches of size ``batch_size`` as usual.

    Each index appears exactly once across all minibatches per epoch — no
    repeated gradients, no IS-ratio explosion.

    How crisis detection works
    --------------------------
    Thresholds ``regime_vol`` (obs index 2816, the first regime feature at the
    last timestep) at the 75th percentile of the current rollout buffer.

    Usage
    -----
        model = PPO(
            policy=MoEActorCriticPolicy,
            env=train_env,
            rollout_buffer_class=RegimeWeightedRolloutBuffer,
            rollout_buffer_kwargs={"crisis_oversample": 3.0},
            ...
        )
    """

    def __init__(self, *args, crisis_oversample: float = 3.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.crisis_oversample = crisis_oversample

    def _compute_sample_weights(self) -> np.ndarray:
        """
        Return per-step sampling weights (not normalised).
        Called BEFORE swap_and_flatten, so observations is still 3-D.
        Shape: (buffer_size * n_envs,)
        """
        obs = self.observations
        if obs.ndim == 3:
            regime_vol = obs[:, :, REGIME_VOL_IDX].flatten()
        else:
            regime_vol = obs[:, REGIME_VOL_IDX]

        threshold   = np.percentile(regime_vol, 75)
        crisis_mask = regime_vol > threshold

        weights             = np.ones(len(regime_vol), dtype=np.float64)
        weights[crisis_mask] = self.crisis_oversample
        return weights

    @staticmethod
    def _weighted_permutation(weights: np.ndarray) -> np.ndarray:
        """
        Gumbel-max trick: produces a random permutation biased toward
        high-weight indices, with each index appearing exactly once.
        """
        log_weights = np.log(weights / weights.sum() + 1e-12)
        gumbel_noise = -np.log(-np.log(np.random.uniform(size=len(weights)) + 1e-12) + 1e-12)
        return np.argsort(-(log_weights + gumbel_noise))

    def get(self, batch_size: Optional[int] = None):
        """
        Yield minibatches with crisis steps biased to appear first.
        Each step appears exactly once per call (PPO-safe).
        """
        assert self.full, "Buffer must be full before sampling."

        n_total = self.buffer_size * self.n_envs

        # Compute weights BEFORE swap_and_flatten (obs still 3-D here)
        weights = self._compute_sample_weights()

        if not self.generator_ready:
            _tensor_names = [
                "observations", "actions", "values", "log_probs",
                "advantages", "returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = n_total

        # Weighted permutation — each index appears exactly once
        indices = self._weighted_permutation(weights)

        start_idx = 0
        while start_idx < n_total:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4.1 — Asymmetric Reward Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AsymmetricRewardWrapper(gym.Wrapper):
    """
    Applies a downside penalty multiplier to the step reward whenever the
    portfolio return is negative (Sortino-inspired asymmetric reward).

    Usage
    -----
        env = CommodityTradingEnv(df, features, config)
        env = AsymmetricRewardWrapper(env, downside_penalty=2.0)

    Parameters
    ----------
    downside_penalty : float
        Multiplier applied to negative rewards.  Default 2.0 means losses
        hurt twice as much as equivalent gains.  Include in Optuna search
        range [1.5, 3.0].
    """

    def __init__(self, env: gym.Env, downside_penalty: float = 2.0) -> None:
        super().__init__(env)
        if downside_penalty < 1.0:
            raise ValueError("downside_penalty must be >= 1.0")
        self.downside_penalty = downside_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward < 0:
            reward = reward * self.downside_penalty
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3.1 — Additional Regime Feature Computation
# ─────────────────────────────────────────────────────────────────────────────

def vol_of_vol(returns: "pd.Series", window: int = 20) -> "pd.Series":
    """
    Rate of change of realised volatility — leads regime transitions.

    Returns the 5-day percentage change in the rolling `window`-day std.
    No lookahead: all values use only past data.
    """
    import pandas as pd
    rolling_vol = returns.rolling(window, min_periods=max(1, window // 2)).std()
    vov = rolling_vol.diff(5) / (rolling_vol.shift(5) + 1e-10)
    return vov.fillna(0.0).clip(-5.0, 5.0)


def regime_transition_speed(
    regime_signals_df: "pd.DataFrame",
    window: int = 10,
) -> "pd.Series":
    """
    Mean absolute rate-of-change of ALL regime signals over `window` days.
    High value → regime transition is underway.

    regime_signals_df : DataFrame with columns [regime_vol, regime_trend,
                         regime_momentum, regime_curve]
    """
    return (
        regime_signals_df
        .diff()
        .abs()
        .rolling(window, min_periods=1)
        .mean()
        .mean(axis=1)
        .fillna(0.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return {layer_name: param_count} and total."""
    counts: Dict[str, int] = {}
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        counts[name] = n
        total += n
    counts["__total__"] = total
    return counts


def make_moe_ppo(
    env,
    total_timesteps: int           = 500_000,
    learning_rate: float           = 3e-6,
    n_steps: int                   = 1024,
    batch_size: int                = 64,
    n_epochs: int                  = 5,
    gamma: float                   = 0.9662,
    gae_lambda: float              = 0.9589,
    clip_range: float              = 0.2,
    vf_coef: float                 = 0.5,
    max_grad_norm: float           = 0.997,
    # Entropy schedule starting value (annealed by EntropyScheduleCallback)
    ent_coef: float                = 0.01,
    # MoE-specific
    n_experts: int                 = N_EXPERTS,
    expert_hidden: int             = 32,
    router_ent_coef: float         = 0.1,
    gate_ent_coef: float           = 0.05,   # encourages flat option usage
    downside_penalty: float        = 2.0,
    crisis_oversample: float       = 3.0,
    use_asymmetric_reward: bool    = True,
    use_regime_buffer: bool        = False,   # opt-in: enable only after HPO confirms stability
    device: str                    = "auto",
    seed: int                      = 42,
    verbose: int                   = 1,
) -> PPO:
    """
    Convenience factory that wires up the MoE PPO model with all v2 components.

    Stability rationale for default hyperparameters
    ------------------------------------------------
    The key PPO stability metric is *per-sample update count* (= n_epochs), NOT
    total gradient steps per rollout.  Each reuse of a sample moves π_new further
    from π_old, driving the IS ratio exp(Δlog_prob) away from 1.0:

      Old settings  n_steps=256,  n_epochs=12: per-sample reuse = 12×
      New settings  n_steps=1024, n_epochs=5 : per-sample reuse = 5×   ← 2.4× safer

    The log_std hard floor (-1.0, set inside MoEActorCriticPolicy) prevents entropy
    falling below 0.42 nats regardless of how many gradient updates fire.

    Example
    -------
        from src.ara_ppo_v2 import make_moe_ppo, AsymmetricRewardWrapper
        from src.callbacks_v2 import make_v2_callbacks

        train_env = CommodityTradingEnv(train_df, features, config)
        if use_asymmetric_reward:
            train_env = AsymmetricRewardWrapper(train_env, downside_penalty=2.0)

        model = make_moe_ppo(train_env, total_timesteps=500_000)
        callbacks = make_v2_callbacks(model, total_timesteps=500_000)
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    """
    if use_asymmetric_reward and not isinstance(env, AsymmetricRewardWrapper):
        env = AsymmetricRewardWrapper(env, downside_penalty=downside_penalty)

    # ── Fix 2 support: infer actual n_features from the env's observation space ──
    # When HMM regime features are added (47 → 51), the CNN encoder must be
    # built with the correct feature count.  obs_dim = N_TIMESTEPS * n_features
    # + N_PORTFOLIO, so n_features = (obs_dim − N_PORTFOLIO) / N_TIMESTEPS.
    _obs_dim = env.observation_space.shape[0]
    _n_features_actual = (_obs_dim - N_PORTFOLIO) // N_TIMESTEPS
    if _n_features_actual <= 0:
        raise ValueError(
            f"Cannot infer n_features from obs_dim={_obs_dim}; "
            f"check N_TIMESTEPS={N_TIMESTEPS}, N_PORTFOLIO={N_PORTFOLIO}"
        )

    policy_kwargs: Dict[str, Any] = dict(
        n_experts       = n_experts,
        expert_hidden   = expert_hidden,
        router_ent_coef = router_ent_coef,
        gate_ent_coef   = gate_ent_coef,
        n_features      = _n_features_actual,   # Fix 2: 47 default, 51 with HMM
    )

    buffer_kwargs: Dict[str, Any] = {}
    buffer_class = RegimeWeightedRolloutBuffer if use_regime_buffer else RolloutBuffer
    if use_regime_buffer:
        buffer_kwargs["crisis_oversample"] = crisis_oversample

    model = PPO(
        policy               = MoEActorCriticPolicy,
        env                  = env,
        learning_rate        = learning_rate,
        n_steps              = n_steps,
        batch_size           = batch_size,
        n_epochs             = n_epochs,
        gamma                = gamma,
        gae_lambda           = gae_lambda,
        clip_range           = clip_range,
        ent_coef             = ent_coef,
        vf_coef              = vf_coef,
        max_grad_norm        = max_grad_norm,
        policy_kwargs        = policy_kwargs,
        rollout_buffer_class = buffer_class,
        rollout_buffer_kwargs= buffer_kwargs,
        device               = device,
        seed                 = seed,
        verbose              = verbose,
    )

    return model
