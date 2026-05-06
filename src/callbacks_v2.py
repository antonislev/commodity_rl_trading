"""
ARA-PPO v2 — Training Callbacks
================================
Phase 1.1  EntropyScheduleCallback  — linear entropy annealing + recovery
Phase 1.3  CollapseMonitorCallback  — detects policy collapse early
           RouterMonitorCallback    — logs MoE router weight distribution
           TrainingMetricsCallback  — collects loss / entropy / Sharpe curves
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.1 — Entropy Scheduling
# ─────────────────────────────────────────────────────────────────────────────

def get_entropy_coeff(
    current_step: int,
    total_steps: int,
    policy_entropy: float,
    start: float          = 0.01,
    end: float            = 0.001,
    decay_fraction: float = 0.8,
    min_entropy: float    = 0.5,    # calibrated for 1D Gaussian (init ≈ 1.42 nats)
    recovery_coeff: float = 0.05,   # 10× stronger than before; must compete with policy gradient
) -> float:
    """
    Compute the entropy coefficient for the current training step.

    Calibration note for continuous (Gaussian) action spaces
    ---------------------------------------------------------
    A 1D DiagGaussian at initialisation (log_std=0) has entropy ≈ 1.42 nats.
    Entropy turns zero when log_std ≈ -1.42 and goes negative for more
    deterministic policies.  The thresholds must reflect this scale:

      min_entropy = 0.5  →  triggers recovery when log_std drops below ≈ -0.92
                             (policy is noticeably over-committing but not yet
                             fully deterministic)
      recovery_coeff = 0.05  →  contributes 0.05 × |entropy| to the loss,
                             strong enough to compete with the policy gradient

    Schedule
    --------
    • Steps 0 → decay_steps   : linear decay from `start` to `end`
    • Steps decay_steps → end  : held constant at `end`
    • Emergency recovery       : if policy_entropy < min_entropy, return
                                  recovery_coeff regardless of step

    Parameters
    ----------
    current_step    : current global timestep
    total_steps     : total planned training timesteps
    policy_entropy  : current mean action distribution entropy (nats)
    start           : entropy coeff at step 0 (default 0.01)
    end             : entropy coeff at end of decay (default 0.001)
    decay_fraction  : fraction of total_steps over which to decay (default 0.8)
    min_entropy     : entropy floor below which recovery fires (default 0.5 nats)
    recovery_coeff  : ent_coef used during emergency recovery (default 0.05)
    """
    # Emergency recovery takes priority
    if policy_entropy < min_entropy:
        return recovery_coeff

    decay_steps = int(total_steps * decay_fraction)
    if current_step < decay_steps:
        t = current_step / max(decay_steps, 1)
        return start + (end - start) * t
    return end


class EntropyScheduleCallback(BaseCallback):
    """
    Updates model.ent_coef at every rollout according to the linear decay
    schedule defined by get_entropy_coeff().

    Also fires an emergency recovery if the policy entropy drops below
    `min_entropy` for `patience` consecutive evaluations.

    Parameters
    ----------
    total_timesteps : total training steps (must match model.learn() arg)
    start           : initial entropy coeff (default 0.01)
    end             : final entropy coeff   (default 0.001)
    decay_fraction  : fraction of steps over which to decay (default 0.8)
    min_entropy     : entropy floor that triggers recovery (default 0.5 nats,
                      calibrated for 1D Gaussian; init entropy ≈ 1.42 nats)
    recovery_coeff  : coeff used during recovery (default 0.05)
    eval_freq       : how often (steps) to check entropy (default 1024)
    verbose         : 0 = silent, 1 = log changes, 2 = log every eval
    """

    def __init__(
        self,
        total_timesteps: int,
        start: float          = 0.01,
        end: float            = 0.001,
        decay_fraction: float = 0.8,
        min_entropy: float    = 0.5,    # 1D Gaussian calibration (init ≈ 1.42 nats)
        recovery_coeff: float = 0.05,   # 10× stronger recovery signal
        eval_freq: int        = 1024,
        verbose: int          = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.total_timesteps = total_timesteps
        self.start           = start
        self.end             = end
        self.decay_fraction  = decay_fraction
        self.min_entropy     = min_entropy
        self.recovery_coeff  = recovery_coeff
        self.eval_freq       = eval_freq

        self._last_entropy: float = float("inf")
        self._in_recovery: bool   = False

    # SB3 calls this after each rollout collection (every n_steps * n_envs steps)
    def _on_rollout_end(self) -> None:
        step = self.num_timesteps

        # Retrieve latest policy entropy from SB3's logger
        entropy = self._get_policy_entropy()
        self._last_entropy = entropy

        new_coef = get_entropy_coeff(
            current_step   = step,
            total_steps    = self.total_timesteps,
            policy_entropy = entropy,
            start          = self.start,
            end            = self.end,
            decay_fraction = self.decay_fraction,
            min_entropy    = self.min_entropy,
            recovery_coeff = self.recovery_coeff,
        )

        recovery_triggered = entropy < self.min_entropy
        if recovery_triggered and not self._in_recovery:
            warnings.warn(
                f"[EntropySchedule] Step {step}: policy entropy {entropy:.4f} "
                f"below floor {self.min_entropy:.3f}. "
                f"Raising ent_coef to {self.recovery_coeff}.",
                stacklevel=2,
            )
            self._in_recovery = True
        elif not recovery_triggered:
            self._in_recovery = False

        # Update the model's ent_coef
        old_coef = float(self.model.ent_coef)
        self.model.ent_coef = float(new_coef)

        if self.verbose >= 2 or (self.verbose >= 1 and abs(new_coef - old_coef) > 1e-6):
            print(
                f"[EntropySchedule] step={step:>7d}  "
                f"ent_coef={new_coef:.5f}  "
                f"policy_entropy={entropy:.4f}"
                + (" ⚠ RECOVERY" if recovery_triggered else "")
            )

        # Log to SB3 logger
        self.logger.record("train/entropy_coef_scheduled", new_coef)
        self.logger.record("train/policy_entropy_monitor", entropy)

        # ── Anneal Gumbel temperature for hybrid gate head ────────────────────
        # 1.0 → 0.3 over first 80% of training, then hold at 0.3.
        # Early: soft samples → gate explores long/short/flat freely
        # Late:  sharp samples → gate commits to clear direction decisions
        progress = step / max(self.total_timesteps, 1)
        gumbel_tau = 1.0 - 0.7 * min(progress / 0.8, 1.0)
        try:
            self.model.policy.action_net.gumbel_tau = gumbel_tau
            self.logger.record("gate/gumbel_tau", gumbel_tau)
        except AttributeError:
            pass  # no hybrid gate head — skip silently

    def _on_step(self) -> bool:
        return True

    def _get_policy_entropy(self) -> float:
        """
        Best-effort attempt to read the mean action entropy from SB3's
        internal logger (populated during PPO.train()).
        Falls back to 1.0 (safe: no spurious recovery) if not available.
        """
        try:
            log_dict = self.logger.name_to_value
            if "train/entropy_loss" in log_dict:
                # SB3 logs entropy_loss = -mean(entropy), so entropy = -entropy_loss
                return max(0.0, -float(log_dict["train/entropy_loss"]))
        except Exception:
            pass
        return 1.0  # safe fallback


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.3 — Policy Collapse Monitor
# ─────────────────────────────────────────────────────────────────────────────

class PolicyCollapseMonitor:
    """
    Stateless-ish monitor that checks for three collapse signatures:

    1. Entropy collapse:    policy_entropy < entropy_threshold for
                            `patience` consecutive calls
    2. Action uniformity:   > action_uniformity_threshold fraction of actions
                            in a batch are identical (after discretisation)
    3. Critic/actor loss    critic_loss ≈ 0 while policy_loss is non-trivial
       divergence:          (symptom of critic overfitting)

    Returns (collapsed: bool, diagnostics: dict).
    """

    def __init__(
        self,
        patience: int                      = 10,
        entropy_threshold: float           = 0.3,   # 1D Gaussian: fire when log_std < -1.12
        action_uniformity_threshold: float = 0.95,
        critic_overfit_ratio: float        = 0.01,
    ) -> None:
        self.patience                    = patience
        self.entropy_threshold           = entropy_threshold
        self.action_uniformity_threshold = action_uniformity_threshold
        self.critic_overfit_ratio        = critic_overfit_ratio

        self._low_entropy_streak: int = 0

    def check(
        self,
        policy_entropy: float,
        actions_batch: np.ndarray,
        policy_loss: Optional[float]  = None,
        critic_loss: Optional[float]  = None,
    ) -> Tuple[bool, Dict]:
        """
        Parameters
        ----------
        policy_entropy  : mean entropy of action distribution (nats)
        actions_batch   : 1-D array of raw action values from current rollout
        policy_loss     : (optional) current PPO policy loss
        critic_loss     : (optional) current value-function loss

        Returns
        -------
        (collapsed, diagnostics)
        """
        # ── 1. Entropy collapse ───────────────────────────────────────────────
        if policy_entropy < self.entropy_threshold:
            self._low_entropy_streak += 1
        else:
            self._low_entropy_streak = 0

        entropy_collapse = self._low_entropy_streak >= self.patience

        # ── 2. Action uniformity ──────────────────────────────────────────────
        # Discretise continuous actions to 10 bins for counting
        actions_binned = np.digitize(
            np.asarray(actions_batch, dtype=np.float32),
            bins=np.linspace(-1.0, 1.0, 11),
        )
        counts = np.bincount(actions_binned, minlength=12)
        most_common_frac = counts.max() / max(len(actions_batch), 1)
        action_uniform = most_common_frac > self.action_uniformity_threshold

        # ── 3. Critic/actor divergence ────────────────────────────────────────
        critic_overfit = False
        if policy_loss is not None and critic_loss is not None:
            if abs(policy_loss) > 1e-6:
                ratio = abs(critic_loss) / (abs(policy_loss) + 1e-12)
                critic_overfit = ratio < self.critic_overfit_ratio

        collapsed = entropy_collapse or action_uniform

        diagnostics = {
            "policy_entropy"       : policy_entropy,
            "low_entropy_streak"   : self._low_entropy_streak,
            "action_uniformity"    : float(most_common_frac),
            "entropy_collapse"     : bool(entropy_collapse),
            "action_collapse"      : bool(action_uniform),
            "critic_overfit_flag"  : bool(critic_overfit),
        }
        return collapsed, diagnostics

    def reset(self) -> None:
        self._low_entropy_streak = 0


class CollapseMonitorCallback(BaseCallback):
    """
    Runs PolicyCollapseMonitor after every rollout.

    On collapse detection:
      • Emits a warning.
      • Optionally stops training early (`halt_on_collapse=True`).
      • Stores collapse events in `self.collapse_log`.
    """

    def __init__(
        self,
        patience: int                      = 10,
        entropy_threshold: float           = 0.3,   # calibrated for 1D Gaussian
        action_uniformity_threshold: float = 0.95,
        halt_on_collapse: bool             = False,
        verbose: int                       = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.monitor = PolicyCollapseMonitor(
            patience                    = patience,
            entropy_threshold           = entropy_threshold,
            action_uniformity_threshold = action_uniformity_threshold,
        )
        self.halt_on_collapse  = halt_on_collapse
        self.collapse_log: List[Dict] = []

    def _on_rollout_end(self) -> None:
        entropy       = self._get_metric("train/entropy_loss", negate=True)
        policy_loss   = self._get_metric("train/policy_gradient_loss")
        critic_loss   = self._get_metric("train/value_loss")

        # Grab last batch of actions from the rollout buffer
        try:
            actions = self.model.rollout_buffer.actions.flatten()
        except Exception:
            actions = np.zeros(1)

        collapsed, diag = self.monitor.check(
            policy_entropy = entropy,
            actions_batch  = actions,
            policy_loss    = policy_loss,
            critic_loss    = critic_loss,
        )

        # Log all diagnostics
        for key, val in diag.items():
            self.logger.record(f"collapse/{key}", float(val))

        if collapsed:
            event = {"step": self.num_timesteps, **diag}
            self.collapse_log.append(event)
            msg = (
                f"[CollapseMonitor] ⚠ COLLAPSE DETECTED at step "
                f"{self.num_timesteps}\n"
                f"  entropy={diag['policy_entropy']:.4f}  "
                f"streak={diag['low_entropy_streak']}  "
                f"uniformity={diag['action_uniformity']:.3f}"
            )
            warnings.warn(msg, stacklevel=2)
            if self.verbose >= 1:
                print(msg)
            if self.halt_on_collapse:
                return False  # stops training

        return True

    def _on_step(self) -> bool:
        return True

    def _get_metric(self, key: str, negate: bool = False) -> float:
        try:
            val = float(self.logger.name_to_value.get(key, 0.0))
            return -val if negate else val
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Router Monitor Callback
# ─────────────────────────────────────────────────────────────────────────────

class RouterMonitorCallback(BaseCallback):
    """
    Logs MoE router weight and gate probability statistics after every rollout.

    Router outputs logged (to SB3 logger):
      router/weight_expert_0   mean weight on expert 0 over last batch
      router/weight_expert_1
      router/weight_expert_2
      router/temperature_tau   current τ value
      router/weight_entropy    entropy of mean weight vector (nats)

    Gate outputs logged (v2.2 HybridMoEActionNet):
      gate/p_long              mean P(long)  over last batch
      gate/p_short             mean P(short)
      gate/p_flat              mean P(flat)  ← key metric for choppy-market fix
      gate/entropy             entropy of mean gate probs (log(3)≈1.1 = uniform)

    Full history stored in:
      self.weight_history  — list of (step, router_weights[n_experts])
      self.gate_history    — list of (step, gate_probs[3])  [long, short, flat]
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.weight_history: List[Tuple[int, np.ndarray]] = []
        self.gate_history:   List[Tuple[int, np.ndarray]] = []

    def _on_rollout_end(self) -> None:
        try:
            action_net = self.model.policy.action_net
        except AttributeError:
            return

        # ── Router weights ────────────────────────────────────────────────────
        weights = action_net._last_weights  # (B, n_experts) or None
        if weights is not None:
            mean_w = weights.mean(axis=0)   # (n_experts,)
            self.weight_history.append((self.num_timesteps, mean_w.copy()))

            for i, w in enumerate(mean_w):
                self.logger.record(f"router/weight_expert_{i}", float(w))

            router_entropy = -(mean_w * np.log(mean_w + 1e-8)).sum()
            self.logger.record("router/weight_entropy", float(router_entropy))

            try:
                tau = float(action_net.tau.item())
                self.logger.record("router/temperature_tau", tau)
            except Exception:
                pass

        # ── Gate probabilities (v2.2 HybridMoEActionNet) ─────────────────────
        gate_probs = getattr(action_net, '_last_gate_probs', None)
        if gate_probs is not None:
            mean_g = gate_probs.mean(axis=0)  # (3,) = [P(long), P(short), P(flat)]
            self.gate_history.append((self.num_timesteps, mean_g.copy()))

            labels = ['long', 'short', 'flat']
            for label, p in zip(labels, mean_g):
                self.logger.record(f"gate/p_{label}", float(p))

            gate_entropy = -(mean_g * np.log(mean_g + 1e-8)).sum()
            self.logger.record("gate/entropy", float(gate_entropy))

    def _on_step(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Training Metrics Callback
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMetricsCallback(BaseCallback):
    """
    Collects a detailed training history for post-hoc analysis and ablation.

    Stores per-rollout records in `self.history` (list of dicts):
      step, ent_coef, policy_loss, value_loss, entropy_loss,
      approx_kl, clip_fraction, explained_variance

    Also checks for critic/actor loss divergence (critic near-zero while
    policy_loss is non-trivial) and logs it as a warning flag.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.history: List[Dict] = []

    def _on_rollout_end(self) -> None:
        log = self.logger.name_to_value
        record = {
            "step"               : self.num_timesteps,
            "ent_coef"           : float(self.model.ent_coef),
            "policy_loss"        : float(log.get("train/policy_gradient_loss", 0.0)),
            "value_loss"         : float(log.get("train/value_loss",           0.0)),
            "entropy_loss"       : float(log.get("train/entropy_loss",         0.0)),
            "approx_kl"          : float(log.get("train/approx_kl",            0.0)),
            "clip_fraction"      : float(log.get("train/clip_fraction",        0.0)),
            "explained_variance" : float(log.get("train/explained_variance",   0.0)),
        }
        self.history.append(record)

        if self.verbose >= 2:
            print(
                f"[Metrics] step={record['step']:>7d}  "
                f"pl={record['policy_loss']:+.4f}  "
                f"vl={record['value_loss']:.4f}  "
                f"ent={record['entropy_loss']:.4f}  "
                f"kl={record['approx_kl']:.4f}"
            )

    def _on_step(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build all v2 callbacks in one call
# ─────────────────────────────────────────────────────────────────────────────

def make_v2_callbacks(
    model,
    total_timesteps: int,
    # Entropy schedule
    entropy_start: float          = 0.01,
    entropy_end: float            = 0.001,
    entropy_decay_fraction: float = 0.8,
    min_entropy_floor: float      = 0.5,    # 1D Gaussian calibration
    recovery_coeff: float         = 0.05,   # 10× stronger recovery
    # Collapse monitor
    collapse_patience: int        = 10,
    entropy_collapse_threshold: float = 0.3,   # 1D Gaussian calibration
    action_uniformity_threshold: float = 0.95,
    halt_on_collapse: bool        = False,
    verbose: int                  = 1,
) -> CallbackList:
    """
    Returns a CallbackList containing all ARA-PPO v2 callbacks.

    Usage
    -----
        callbacks = make_v2_callbacks(model, total_timesteps=500_000)
        model.learn(total_timesteps=500_000, callback=callbacks)

    Access post-training data
    -------------------------
        cb_list = callbacks.callbacks
        entropy_cb = cb_list[0]    # EntropyScheduleCallback
        collapse_cb = cb_list[1]   # CollapseMonitorCallback
        router_cb  = cb_list[2]    # RouterMonitorCallback
        metrics_cb = cb_list[3]    # TrainingMetricsCallback

        import pandas as pd
        history_df = pd.DataFrame(metrics_cb.history)
        router_df  = pd.DataFrame(
            [(s, *w) for s, w in router_cb.weight_history],
            columns=["step", "expert_0", "expert_1", "expert_2"]
        )
    """
    entropy_cb = EntropyScheduleCallback(
        total_timesteps  = total_timesteps,
        start            = entropy_start,
        end              = entropy_end,
        decay_fraction   = entropy_decay_fraction,
        min_entropy      = min_entropy_floor,
        recovery_coeff   = recovery_coeff,
        verbose          = verbose,
    )

    collapse_cb = CollapseMonitorCallback(
        patience                    = collapse_patience,
        entropy_threshold           = entropy_collapse_threshold,
        action_uniformity_threshold = action_uniformity_threshold,
        halt_on_collapse            = halt_on_collapse,
        verbose                     = verbose,
    )

    router_cb  = RouterMonitorCallback(verbose=verbose)
    metrics_cb = TrainingMetricsCallback(verbose=0)

    return CallbackList([entropy_cb, collapse_cb, router_cb, metrics_cb])