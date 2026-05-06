"""
ARA-PPO v2.3 — HMM Regime Detector
===================================
Gaussian Hidden Markov Model fit per walk-forward split (no lookahead).

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
        n_iter: int       = 500,    # was 100 — give EM enough iterations
        tol: float        = 1e-4,   # explicit convergence tolerance
        random_state: int = 42,
    ) -> None:
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required.  Install with: pip install hmmlearn"
            )
        self.n_states     = n_states
        self.vol_window   = vol_window
        self.n_iter       = n_iter
        self.tol          = tol
        self.random_state = random_state
        self.hmm: Optional[GaussianHMM] = None
        self._fitted: bool = False

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        """Build (T, 2) input matrix: [log_return, rolling_vol_20d]."""
        ret = df["log_return"].fillna(0).values.astype(np.float64)
        vol = (
            df["log_return"]
            .rolling(self.vol_window, min_periods=1)
            .std()
            .fillna(0)
            .values
            .astype(np.float64)
        )
        return np.column_stack([ret, vol])

    def fit(self, df: pd.DataFrame) -> "RegimeHMM":
        """Fit HMM on the training-window dataframe."""
        X = self._features(df)
        self.hmm = GaussianHMM(
            n_components    = self.n_states,
            covariance_type = "diag",
            n_iter          = self.n_iter,
            tol             = self.tol,
            random_state    = self.random_state,
            init_params     = "stmc",   # init starts/transmat/means/covars
            verbose         = False,
        )
        # Silence hmmlearn's ConvergenceMonitor (uses bare print, not warnings)
        # plus the warnings module for safety.
        import warnings, sys, io, contextlib
        buf = io.StringIO()
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(buf), \
             contextlib.redirect_stderr(buf):
            warnings.simplefilter("ignore")
            self.hmm.fit(X)
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
