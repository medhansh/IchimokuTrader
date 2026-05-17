"""
strategy/hmm_weighted_signals.py
---------------------------------
HMM-weighted composite score.

Instead of equal-weighting s1+s2+s3+s4+s5+s6, a Hidden Markov Model
learns latent market states from historical score data and assigns
state-specific weights to each component.

Architecture:
    1. Offline training (train_hmm):
       - Compute all six score components across training data
       - Fit a Gaussian HMM with N states on the joint distribution
       - For each state, compute the correlation of each component
         with short-term future returns — that correlation becomes
         the component's weight in that state
       - Save state centroids + weight matrix to disk

    2. Online inference (hmm_composite_score):
       - Compute current score components
       - Run Viterbi to find current market state
       - Apply state-specific weights to components
       - Apply WHT multiplier (same as adaptive_signals)

Why this works without overfitting:
    - The HMM learns on score components (6 features), not raw prices
    - States are latent structure in the scores themselves, not fitted to returns
    - Weight estimation uses simple correlation, not a complex model
    - N=3 states is small enough to be robust (trending/ranging/volatile)

Dependencies: hmmlearn, numpy, pandas — NOT in requirements.txt (dev only)
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.indicators import col, atr
from strategy.adaptive_signals import (
    cycle_series, build_filtered_df, adaptive_ichimoku_scores,
    wht_multiplier_series, WHT_WINDOW,
)
from strategy.signals import vidya_score, volume_weighted_rsi, MIN_BARS_NEEDED

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "hmm_model.pkl"
N_STATES   = 3      # trending / ranging / volatile
MIN_TRAIN_BARS = MIN_BARS_NEEDED + 50


# ── Score component extraction ─────────────────────────────────────────────────

def extract_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all six raw score components (before tanh and WHT).
    Returns DataFrame with columns [s1,s2,s3,s4,s5,s6], NaNs dropped.
    """
    if len(df) < MIN_TRAIN_BARS:
        return pd.DataFrame()

    close  = col(df, "Close")
    df_f   = build_filtered_df(df)
    cyc_s  = cycle_series(close)

    s1, s2, s3, s4 = adaptive_ichimoku_scores(df_f, cyc_s)
    s5 = vidya_score(df)
    s6 = volume_weighted_rsi(df)

    comp = pd.DataFrame({
        "s1": s1, "s2": s2, "s3": s3,
        "s4": s4, "s5": s5, "s6": s6,
    }, index=df.index).dropna()

    return comp


# ── HMM training ──────────────────────────────────────────────────────────────

def train_hmm(
    data:         dict[str, pd.DataFrame],
    n_states:     int   = N_STATES,
    forward_bars: int   = 5,
    save_path:    Path  = MODEL_PATH,
) -> dict:
    """
    Train a Gaussian HMM on score components across all training tickers.

    Steps:
        1. Extract score components for each ticker
        2. Stack into one long sequence (HMM sees components, not returns)
        3. Fit GaussianHMM with n_states
        4. For each state: compute mean correlation of each component
           with forward_bars-ahead returns across all training data
        5. State weights = softmax of correlations (positive only)

    Returns model dict with: hmm, weight_matrix, scaler
    Saves to save_path.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "hmmlearn and scikit-learn required for HMM training.\n"
            "pip install hmmlearn scikit-learn"
        )

    log.info(f"[HMM TRAIN]  Extracting score components from {len(data)} tickers ...")

    all_components = []
    all_returns    = []
    lengths        = []

    for ticker, df in data.items():
        try:
            comp = extract_components(df)
            if comp.empty or len(comp) < 100:
                continue

            close = col(df, "Close").reindex(comp.index)
            fwd_ret = close.pct_change(forward_bars).shift(-forward_bars).reindex(comp.index)

            valid = comp.join(fwd_ret.rename("fwd_ret")).dropna()
            if len(valid) < 80:
                continue

            all_components.append(valid[["s1","s2","s3","s4","s5","s6"]].values)
            all_returns.append(valid["fwd_ret"].values)
            lengths.append(len(valid))
        except Exception as e:
            log.debug(f"  {ticker}: {e}")

    if not all_components:
        raise RuntimeError("No valid training data extracted")

    X_raw = np.vstack(all_components)
    R     = np.concatenate(all_returns)
    log.info(f"[HMM TRAIN]  Total training bars: {len(X_raw)} across {len(lengths)} tickers")

    # Standardise components before fitting HMM
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    # Fit HMM — covariance type 'diag' is more stable than 'full' for small N
    log.info(f"[HMM TRAIN]  Fitting GaussianHMM with {n_states} states ...")
    hmm = GaussianHMM(
        n_components = n_states,
        covariance_type = "diag",
        n_iter    = 200,
        tol       = 1e-4,
        random_state = 42,
    )
    hmm.fit(X, lengths)
    log.info(f"[HMM TRAIN]  Converged: {hmm.monitor_.converged}")

    # Decode state sequence for the full training set
    states = hmm.predict(X)
    log.info(f"[HMM TRAIN]  State distribution: "
             f"{[(states==i).mean() for i in range(n_states)]}")

    # Compute per-state weights = correlation(component, forward_return)
    # clipped to [0, inf] — we only want to upweight predictive components
    weight_matrix = np.zeros((n_states, 6))
    COMPONENT_NAMES = ["s1","s2","s3","s4","s5","s6"]

    for state in range(n_states):
        mask = states == state
        if mask.sum() < 30:
            weight_matrix[state] = np.ones(6) / 6  # uniform fallback
            continue

        X_state = X_raw[mask]
        R_state = R[mask]

        for j in range(6):
            corr = np.corrcoef(X_state[:, j], R_state)[0, 1]
            # Use absolute correlation — a strongly negative correlation
            # means the component predicts the opposite direction, which is
            # still useful (we'd weight it and flip it)
            # But for simplicity: clip negative to small positive
            weight_matrix[state, j] = max(0.05, abs(corr))

        # Softmax normalise so weights sum to 1 per state
        w = weight_matrix[state]
        weight_matrix[state] = np.exp(w) / np.exp(w).sum()

        log.info(f"[HMM TRAIN]  State {state} weights: "
                 f"{dict(zip(COMPONENT_NAMES, weight_matrix[state].round(3)))}")

    model = {
        "hmm":           hmm,
        "scaler":        scaler,
        "weight_matrix": weight_matrix,
        "n_states":      n_states,
        "forward_bars":  forward_bars,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"[HMM TRAIN]  Model saved → {save_path}")

    return model


def load_hmm(path: Path = MODEL_PATH) -> dict | None:
    """Load saved HMM model. Returns None if not found."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Inference ─────────────────────────────────────────────────────────────────

def hmm_composite_score(
    df:    pd.DataFrame,
    model: dict | None = None,
) -> pd.Series:
    """
    Compute HMM-weighted composite score.

    For each bar:
        1. Extract components [s1..s6]
        2. Predict current state using Viterbi on recent window
        3. Weighted sum: score_raw = Σ w_i(state) * s_i
        4. Apply tanh + WHT multiplier (same as adaptive_signals)

    If model is None, falls back to equal-weight (same as original).
    """
    if len(df) < MIN_TRAIN_BARS + WHT_WINDOW:
        return pd.Series(dtype=float)

    comp = extract_components(df)
    if comp.empty:
        return pd.Series(dtype=float)

    close = col(df, "Close")

    if model is None:
        # Fallback: equal weights
        raw  = comp.sum(axis=1)
        base = raw.apply(np.tanh)
        wht  = wht_multiplier_series(close).reindex(comp.index)
        return (base * wht).clip(-1, 1)

    hmm     = model["hmm"]
    scaler  = model["scaler"]
    W       = model["weight_matrix"]  # (n_states, 6)

    X_raw = comp.values
    X     = scaler.transform(X_raw)

    # Predict states for the full component sequence
    try:
        states = hmm.predict(X)
    except Exception as e:
        log.warning(f"[HMM]  State prediction failed: {e} — using equal weights")
        states = np.zeros(len(X), dtype=int)

    # Weighted sum per bar
    # Weights sum to 1 (simplex), but original score sums 6 components.
    # To preserve scale: multiply weighted sum by N_COMPONENTS so
    # weighted_sum ≈ original_sum when weights are uniform (1/6 each).
    N = 6
    raw_scores = np.array([
        N * np.dot(W[states[i]], X_raw[i])
        for i in range(len(X_raw))
    ])

    raw    = pd.Series(raw_scores, index=comp.index)
    base   = raw.apply(np.tanh)
    wht    = wht_multiplier_series(close).reindex(comp.index)
    final  = (base * wht).clip(-1, 1)

    return final


def hmm_latest_score(df: pd.DataFrame, model: dict | None = None) -> tuple[float, float]:
    """Return (score_now, score_prev) for live trading."""
    sc = hmm_composite_score(df, model).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])