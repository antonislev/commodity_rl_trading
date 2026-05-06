
# src/environment.py
# Commodity Trading Environment for ARA-PPO
# Antonis Leveidiotis | University of Piraeus

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import RobustScaler


class CommodityTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, features, config, mode='train', scaler=None, render_mode=None):
        super().__init__()
        self.df = df.copy()
        self.features = features
        self.config = config
        self.mode = mode
        self.render_mode = render_mode
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f'Missing features: {missing}')
        self.n_features = len(features)
        self.lookback = config['lookback_window']
        self.episode_length = config['episode_length']
        self.n_rows = len(df)
        self.min_rows = self.lookback + self.episode_length
        if self.n_rows < self.min_rows:
            raise ValueError(f'Need {self.min_rows} rows, got {self.n_rows}')
        if mode == 'train' or scaler is None:
            self.scaler = RobustScaler()
            self.scaler.fit(df[features].values)
        else:
            self.scaler = scaler
        scaled = self.scaler.transform(df[features].values)
        self.features_arr = np.clip(scaled, -config['clip_obs'], config['clip_obs']).astype(np.float32)
        self.close_arr  = df['close'].values.astype(np.float32)
        self.return_arr = df['log_return'].fillna(0).values.astype(np.float32)
        obs_dim = self.lookback * self.n_features + 5
        self.observation_space = spaces.Box(
            low=-config['clip_obs'], high=config['clip_obs'],
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([config['action_min']], dtype=np.float32),
            high=np.array([config['action_max']], dtype=np.float32),
            shape=(1,), dtype=np.float32
        )
        self._init_state()

    def _init_state(self):
        self.current_step = 0
        self.start_idx = 0
        self.portfolio_value = float(self.config['initial_capital'])
        self.cash = self.portfolio_value
        self.position = 0.0
        self.peak_value = self.portfolio_value
        self.total_costs = 0.0
        self.trade_count = 0
        self.return_history = []
        self.value_history  = [self.portfolio_value]
        self.action_history = []
        self.reward_history = []
        # Fix 1: differential Sharpe ratio EMA state (Moody-Saffell 2001)
        # A_t = EMA of return,  B_t = EMA of squared return
        self.dsr_A = 0.0
        self.dsr_B = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self.n_rows - self.episode_length - self.lookback
        if self.mode == 'train':
            self.start_idx = int(self.np_random.integers(0, max(1, max_start)))
        else:
            self.start_idx = 0
        self._init_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        old_pos = self.position  # capture before update — needed for direction-change penalty
        new_pos = float(np.clip(action[0], self.config['action_min'], self.config['action_max']))
        max_change = self.config['position_change_limit']
        delta = np.clip(new_pos - old_pos, -max_change, max_change)
        new_pos = old_pos + delta
        cost = self._compute_cost(old_pos, new_pos)
        data_idx = self.start_idx + self.lookback + self.current_step
        market_ret = float(self.return_arr[min(data_idx, len(self.return_arr)-1)])
        port_ret = old_pos * market_ret - cost
        new_val = self.portfolio_value * (1 + port_ret)
        self.position = new_pos
        self.portfolio_value = new_val
        self.peak_value = max(self.peak_value, new_val)
        self.total_costs += cost * self.portfolio_value
        self.trade_count += 1 if abs(delta) > 0.01 else 0
        reward = self._compute_reward(port_ret, cost, old_pos)
        self.return_history.append(port_ret)
        self.value_history.append(self.portfolio_value)
        self.action_history.append(new_pos)
        self.reward_history.append(reward)
        self.current_step += 1
        truncated  = self.current_step >= self.episode_length
        terminated = self._check_termination()
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _compute_cost(self, old_pos, new_pos):
        trade_size = abs(new_pos - old_pos)
        if trade_size < 1e-6:
            return 0.0
        cost = trade_size * (
            self.config['commission_per_trade'] +
            self.config['bid_ask_spread'] +
            self.config['slippage']
        )
        if self.current_step % 21 == 0 and abs(new_pos) > 0.01:
            cost += abs(new_pos) * self.config['roll_cost']
        return float(cost)

    def _compute_reward(self, port_ret, cost, old_position=None):
        """
        Dispatcher for reward computation.

        config['reward_type']:
          'composite'           : original 4-term weighted reward (default)
          'differential_sharpe' : Moody-Saffell DSR (Fix 1)
        """
        cfg = self.config
        reward_type = cfg.get('reward_type', 'composite')
        if reward_type == 'differential_sharpe':
            return self._reward_dsr(port_ret, cost, old_position)
        return self._reward_composite(port_ret, cost, old_position)

    def _reward_composite(self, port_ret, cost, old_position=None):
        """Original 4-term weighted reward (kept for backward compatibility)."""
        cfg = self.config
        r_return = port_ret
        drawdown = self.portfolio_value / self.peak_value - 1 if self.peak_value > 0 else 0.0
        r_drawdown = min(drawdown, 0.0)
        r_transaction = -cost
        r_sharpe = 0.0
        w = cfg['sharpe_window']
        if len(self.return_history) >= w:
            rets = np.array(self.return_history[-w:])
            r_sharpe = np.clip(
                rets.mean() / (rets.std() + 1e-10) * np.sqrt(252/w),
                -5.0, 5.0
            )
        reward = (
            cfg['w_return']      * r_return +
            cfg['w_drawdown']    * r_drawdown +
            cfg['w_transaction'] * r_transaction +
            cfg['w_sharpe']      * r_sharpe * 0.01
        ) * cfg['reward_scaling']
        # Phase 4.1: asymmetric downside penalty (Sortino-inspired).
        asym = cfg.get('asymmetric_penalty', 1.0)
        if asym != 1.0 and reward < 0:
            reward = reward * asym
        # v2.2: direction-change penalty — deters long→short whipsaw in choppy markets.
        dcp = cfg.get('direction_change_penalty', 0.0)
        if dcp > 0.0 and old_position is not None:
            if (abs(old_position) > 0.05 and abs(self.position) > 0.05
                    and np.sign(old_position) != np.sign(self.position)):
                reward -= dcp
        return float(reward)

    def _reward_dsr(self, port_ret, cost, old_position=None):
        """
        Fix 1: Differential Sharpe Ratio reward (Moody & Saffell, 2001).

        Reward = D_t  −  λ_cost · cost  −  λ_turnover · 1{trade}  −  dcp · 1{flip}

        D_t is the marginal contribution of return r_t to the running Sharpe:

            A_t = A_{t-1} + η · (r_t − A_{t-1})        (EMA of return)
            B_t = B_{t-1} + η · (r_t² − B_{t-1})       (EMA of squared return)
            D_t = (B_{t-1}·Δ_A − 0.5·A_{t-1}·Δ_B) / (B_{t-1} − A_{t-1}²)^{3/2}

        Properties
        ----------
        - **Scale-invariant in position size**: doubling positions doubles
          numerator and denominator → D unchanged. Fixes the under-leverage
          problem of additive reward shaping.
        - **Online**: no episode-window mismatch; updates every step.
        - **Stationary**: variance term self-normalises the return signal.

        Numerical safety
        ----------------
        - Variance floor (1e-6) prevents div-by-zero on flat return runs
        - Output clipped to [-10, +10] to bound reward magnitude
        - First few steps (A=B=0) produce D≈0 — natural warmup, no special case
        """
        cfg = self.config
        eta             = cfg.get('dsr_eta',         0.01)
        lambda_cost     = cfg.get('lambda_cost',     1.0)
        lambda_turnover = cfg.get('lambda_turnover', 0.0)

        A_prev  = self.dsr_A
        B_prev  = self.dsr_B
        delta_A = port_ret    - A_prev
        delta_B = port_ret**2 - B_prev

        var_prev = max(B_prev - A_prev**2, 1e-6)
        D = (B_prev * delta_A - 0.5 * A_prev * delta_B) / (var_prev ** 1.5)
        D = float(np.clip(D, -10.0, 10.0))

        # Update EMA state for next step
        self.dsr_A = A_prev + eta * delta_A
        self.dsr_B = B_prev + eta * delta_B

        reward = D - lambda_cost * cost

        # Turnover penalty (binary trade indicator)
        if old_position is not None and lambda_turnover > 0.0:
            delta_pos = abs(self.position - old_position)
            if delta_pos > 0.01:
                reward -= lambda_turnover

        # v2.2 direction-change penalty (kept; reward-scale independent)
        dcp = cfg.get('direction_change_penalty', 0.0)
        if dcp > 0.0 and old_position is not None:
            if (abs(old_position) > 0.05 and abs(self.position) > 0.05
                    and np.sign(old_position) != np.sign(self.position)):
                reward -= dcp

        return float(reward)

    def _check_termination(self):
        if self.peak_value <= 0:
            return True
        return (self.portfolio_value / self.peak_value - 1) < self.config['max_drawdown_limit']

    def _get_observation(self):
        start = self.start_idx + self.current_step
        end   = start + self.lookback
        end   = min(end, len(self.features_arr))
        start = max(0, end - self.lookback)
        feat_window = self.features_arr[start:end]
        if len(feat_window) < self.lookback:
            pad = np.zeros((self.lookback - len(feat_window), self.n_features), dtype=np.float32)
            feat_window = np.vstack([pad, feat_window])
        drawdown = self.portfolio_value / self.peak_value - 1 if self.peak_value > 0 else 0.0
        log_pnl  = np.log(self.portfolio_value / self.config['initial_capital'] + 1e-10)
        port_state = np.array([
            self.position,
            np.clip(drawdown, -1.0, 0.0),
            np.clip(log_pnl, -5.0, 5.0),
            self.current_step / self.episode_length,
            np.clip(self.total_costs / self.config['initial_capital'], 0, 0.1)
        ], dtype=np.float32)
        return np.concatenate([feat_window.flatten(), port_state]).astype(np.float32)

    def _get_info(self):
        dd = self.portfolio_value / self.peak_value - 1 if self.peak_value > 0 else 0.0
        tr = self.portfolio_value / self.config['initial_capital'] - 1
        sharpe = 0.0
        if len(self.return_history) >= self.config['sharpe_window']:
            rets = np.array(self.return_history)
            sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'total_return': tr,
            'drawdown': dd,
            'sharpe': sharpe,
            'total_costs': self.total_costs,
            'trade_count': self.trade_count
        }

    def get_episode_metrics(self):
        values  = np.array(self.value_history)
        returns = np.array(self.return_history)
        if len(returns) < 2:
            return {}
        total_return = values[-1] / values[0] - 1
        n_days    = len(returns)
        ann_ret   = (1 + total_return) ** (252 / n_days) - 1
        ann_vol   = returns.std() * np.sqrt(252)
        sharpe    = ann_ret / (ann_vol + 1e-10)
        peak      = np.maximum.accumulate(values)
        dd        = (values - peak) / (peak + 1e-10)
        max_dd    = dd.min()
        calmar    = ann_ret / (abs(max_dd) + 1e-10)
        neg_rets  = returns[returns < 0]
        down_std  = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 1e-10
        sortino   = ann_ret / (down_std + 1e-10)
        win_rate  = (returns > 0).mean()
        return {
            'total_return'  : float(total_return),
            'ann_return'    : float(ann_ret),
            'ann_vol'       : float(ann_vol),
            'sharpe'        : float(sharpe),
            'max_drawdown'  : float(max_dd),
            'calmar'        : float(calmar),
            'sortino'       : float(sortino),
            'win_rate'      : float(win_rate),
            'total_costs'   : float(self.total_costs),
            'trade_count'   : int(self.trade_count),
            'episode_length': int(len(returns))
        }

    def render(self):
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']:4d} | Value: ${info['portfolio_value']:10.2f} | "
                  f"Pos: {info['position']:+.3f} | Return: {info['total_return']:+.4f}")

    def close(self):
        pass


class WalkForwardEnvFactory:
    def __init__(self, df, features, splits, config):
        self.df = df
        self.features = features
        self.splits = splits
        self.config = config

    def get_split(self, split_idx):
        split = self.splits[split_idx]
        train_df = self.df[
            (self.df.index >= split['train_start']) &
            (self.df.index <  split['train_end'])
        ].copy()
        test_df = self.df[
            (self.df.index >= split['test_start']) &
            (self.df.index <  split['test_end'])
        ].copy()

        # Use a local feature list — HMM features are split-specific so we don't
        # mutate self.features.
        features_used = list(self.features)

        # ── Fix 2: HMM regime features (fit on TRAIN only, predict on both) ──
        # No lookahead: hmm.fit() sees train_df; predict_proba is then applied
        # to both windows.  Each split refits a fresh HMM.
        if self.config.get('use_hmm_regimes', False):
            from src.regime_hmm import RegimeHMM
            n_states = self.config.get('hmm_states', 4)
            hmm = RegimeHMM(
                n_states     = n_states,
                vol_window   = self.config.get('hmm_vol_window', 20),
                random_state = self.config.get('hmm_seed', 42),
            ).fit(train_df)

            train_probs = hmm.predict_proba(train_df)   # (T_train, K)
            test_probs  = hmm.predict_proba(test_df)    # (T_test,  K)

            for i in range(n_states):
                col = f'regime_hmm_{i}'
                train_df[col] = train_probs[:, i]
                test_df[col]  = test_probs[:, i]
                features_used.append(col)

        scaler = RobustScaler()
        scaler.fit(train_df[features_used].values)
        test_config  = dict(self.config)
        train_config = dict(self.config)
        min_test_ep  = len(test_df) - self.config['lookback_window']
        min_train_ep = len(train_df) - self.config['lookback_window']
        if min_test_ep < self.config['episode_length']:
            if min_test_ep < 20:
                return None, None, None
            test_config['episode_length'] = min_test_ep
        if min_train_ep < self.config['episode_length']:
            train_config['episode_length'] = min_train_ep
        train_env = CommodityTradingEnv(train_df, features_used, train_config, 'train', scaler)
        test_env  = CommodityTradingEnv(test_df,  features_used, test_config,  'test',  scaler)
        return train_env, test_env, scaler

    def n_splits(self):
        return len(self.splits)


class BaselineStrategies:
    @staticmethod
    def buy_and_hold(env):
        env.reset(seed=42)
        while True:
            _, _, t, tr, _ = env.step(np.array([1.0]))
            if t or tr: break
        return env.get_episode_metrics()

    @staticmethod
    def moving_average_crossover(env, fast='price_to_sma_21', slow='price_to_sma_50'):
        env.reset(seed=42)
        while True:
            idx = min(env.start_idx + env.lookback + env.current_step - 1, len(env.df)-1)
            fv  = env.df.iloc[idx].get(fast, 0) if fast in env.df.columns else 0
            sv  = env.df.iloc[idx].get(slow, 0) if slow in env.df.columns else 0
            pos = 1.0 if fv > sv else -1.0
            _, _, t, tr, _ = env.step(np.array([pos], dtype=np.float32))
            if t or tr: break
        return env.get_episode_metrics()

    @staticmethod
    def random_agent(env, n_runs=10):
        all_m = []
        for s in range(n_runs):
            env.reset(seed=s)
            while True:
                _, _, t, tr, _ = env.step(env.action_space.sample())
                if t or tr: break
            all_m.append(env.get_episode_metrics())
        avg = {k: np.mean([m.get(k, 0) for m in all_m]) for k in all_m[0]}
        return avg

    @staticmethod
    def momentum_strategy(env, col='momentum_1m'):
        env.reset(seed=42)
        while True:
            idx = min(env.start_idx + env.lookback + env.current_step - 1, len(env.df)-1)
            mom = env.df.iloc[idx].get(col, 0) if col in env.df.columns else 0
            pos = float(np.sign(mom)) if mom != 0 else 0.0
            _, _, t, tr, _ = env.step(np.array([pos], dtype=np.float32))
            if t or tr: break
        return env.get_episode_metrics()
