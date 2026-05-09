"""
ARA-PPO v2.3c — HMM Regime Detector with directional features
==============================================================
Gaussian Hidden Markov Model fit per walk-forward split (no lookahead).

Feature set (v2.3c update)
--------------------------
Originally [log_return, rolling_vol_20d] — but the v2.3a run regressed
because the HMM clustered by *magnitude of vol* and lumped the 2009–2010
high-vol recovery rally into the same state as the 2008 crash.  Both have
high vol; only the *direction* of returns distinguishes them.

Added a rolling mean-return feature:  [log_return, rolling_vol_20d,
rolling_mean_return_20d].  The mean-return feature gives the HMM a
directional axis, so up-trends and down-trends become separable states
even when both have similar vol.

Why HMM not K-means?
--------------------
K-means produces noisy per-timestep labels with no temporal model.  HMM
encodes regime persistence via the transition matrix, gives soft posterior
probabilities that can be used directly as features, and aligns with the
canonical financial-econometrics approach (Hamilton 1989).

Usage
-----
    hmm = RegimeHMM(n_states=4).fit(train_df)
    train_probs = hmm.predict_proba(train_df)   # (T_train, n_states)
    test_probs  = hmm.predict_proba(test_df)    # (T_test,  n_states)

The factory must call .fit() on TRAIN data only, then predict on test —
otherwise lookahead bias contaminates the regime feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class _UniformHMMFallback:
    """
    Last-resort fallback when GaussianHMM fit fails on all retries.
    Returns uniform regime probabilities — degenerate but never crashes.
    Same .predict_proba() interface as hmmlearn.GaussianHMM.
    """

    def __init__(self, n_states: int):
        self.n_components = n_states
        self.startprob_   = np.ones(n_states) / n_states
        self.transmat_    = np.ones((n_states, n_states)) / n_states
        self.means_       = np.zeros((n_states, 2))
        self.covars_      = np.ones((n_states, 2))

    def predict_proba(self, X):
        T = len(X)
        return np.ones((T, self.n_components), dtype=np.float64) / self.n_components


class RegimeHMM:
    """
    Gaussian HMM fit on (log_return, rolling_vol) for unsupervised regime
    detection on financial returns.

    Parameters
    ----------
    n_states : int
        Number of latent regimes (typical: 3-4 for daily commodity returns)
    vol_window : int
        Window for rolling volatility feature (default 20 days)
    n_iter : int
        Maximum EM iterations
    random_state : int
        Reproducibility seed
    """

    def __init__(
        self,
        n_states: int     = 4,
        vol_window: int   = 20,
        mean_window: int  = 20,    # v2.3c: rolling mean-return window
        n_iter: int       = 500,
        tol: float        = 1e-4,
        random_state: int = 42,
    ) -> None:
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required.  Install with: pip install hmmlearn"
            )
        self.n_states     = n_states
        self.vol_window   = vol_window
        self.mean_window  = mean_window
        self.n_iter       = n_iter
        self.tol          = tol
        self.random_state = random_state
        self.hmm: Optional[GaussianHMM] = None
        self._fitted: bool = False

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build (T, 3) input matrix: [log_return, rolling_vol, rolling_mean].

        v2.3c: the rolling-mean feature gives the HMM a directional axis so
        up-trending and down-trending high-vol periods become separable
        states.  Without it, the HMM clustered the 2009-2010 recovery into
        the same state as the 2008 crash, causing wrong-way trades.
        """
        ret = df["log_return"].fillna(0).values.astype(np.float64)
        vol = (
            df["log_return"]
            .rolling(self.vol_window, min_periods=1)
            .std()
            .fillna(0)
            .values
            .astype(np.float64)
        )
        mean = (
            df["log_return"]
            .rolling(self.mean_window, min_periods=1)
            .mean()
            .fillna(0)
            .values
            .astype(np.float64)
        )
        return np.column_stack([ret, vol, mean])

    def fit(self, df: pd.DataFrame) -> "RegimeHMM":
        """
        Fit HMM on the training-window dataframe with NaN-safe retry logic.

        Sometimes EM diverges and produces NaN parameters (especially on
        regime-degenerate windows).  We detect this and retry with a different
        random seed up to `n_retries` times.  If all retries fail we fall back
        to a uniform-prior 1-state model that always returns equal regime
        probabilities — ensures downstream code never crashes.
        """
        import warnings as _warn, io, contextlib
        X = self._features(df)
        # Replace any non-finite rows defensively (shouldn't happen but cheap)
        X = np.where(np.isfinite(X), X, 0.0)

        n_retries = 5
        last_err = None
        for attempt in range(n_retries):
            seed = self.random_state + attempt * 7919  # different seed each retry
            self.hmm = GaussianHMM(
                n_components    = self.n_states,
                covariance_type = "diag",
                n_iter          = self.n_iter,
                tol             = self.tol,
                random_state    = seed,
                init_params     = "stmc",
                verbose         = False,
            )
            buf = io.StringIO()
            try:
                with _warn.catch_warnings(), \
                     contextlib.redirect_stdout(buf), \
                     contextlib.redirect_stderr(buf):
                    _warn.simplefilter("ignore")
                    self.hmm.fit(X)
                # Validate fitted parameters are all finite
                ok = (
                    np.all(np.isfinite(self.hmm.startprob_)) and
                    np.all(np.isfinite(self.hmm.transmat_))  and
                    np.all(np.isfinite(self.hmm.means_))     and
                    np.all(np.isfinite(self.hmm.covars_))
                )
                if ok:
                    self._fitted = True
                    return self
            except Exception as e:
                last_err = e

        # Final fallback: build a degenerate model that returns uniform probs.
        # Don't crash — let the ensemble downstream still receive valid input.
        self.hmm = _UniformHMMFallback(self.n_states)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return (T, n_states) array of regime posterior probabilities.
        Sums to 1.0 across each row.
        """
        if not self._fitted:
            raise RuntimeError("RegimeHMM.fit() must be called before predict_proba()")
        X = self._features(df)
        return self.hmm.predict_proba(X).astype(np.float32)

    def regime_means(self) -> np.ndarray:
        """Return (n_states, 2) array of regime means [μ_return, μ_vol]."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return self.hmm.means_

    def transition_matrix(self) -> np.ndarray:
        """Return (n_states, n_states) transition probability matrix."""
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return self.hmm.transmat_
