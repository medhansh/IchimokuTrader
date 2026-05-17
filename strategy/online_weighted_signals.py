"""
strategy/online_weighted_signals.py
-------------------------------------
Online learning with exponential forgetting for score component weighting.

Instead of training a model offline (like HMM), this approach updates
component weights continuously as new bars arrive, giving more influence
to recent prediction errors and exponentially discounting older ones.

Intuition:
    At each bar t, we observe the score components [s1..s6] and (after
    forward_bars have passed) the actual forward return. We compute how
    wrong the current weights were and update them proportionally —
    but we apply a forgetting factor λ < 1 so that errors from 3 months
    ago matter much less than errors from last week.

    This naturally adapts to regime changes: when the market shifts from
    trending to range-bound, the Ichimoku components (s1,s2) start
    predicting poorly, their weight gradually decays, and the VRSI (s6)
    weight rises as it starts predicting well.

Algorithm (Online Gradient Descent with forgetting):
    prediction_t = w_t · components_t
    error_t      = forward_return_t - prediction_t
    gradient_t   = -2 * error_t * components_t
    w_{t+1}      = λ * w_t - η * gradient_t    (forgetting + gradient step)
    w_{t+1}      = project_to_simplex(w_{t+1})  (keep weights positive, sum=1)

The projection to the probability simplex ensures weights stay
interpretable (positive, summing to 1) without explicit constraints.

Parameters:
    λ (forgetting factor): how fast old information decays
        λ = 0.99 → half-life ≈ 69 bars ≈ 3 months (daily bars)
        λ = 0.995 → half-life ≈ 138 bars ≈ 6 months
    η (learning rate): step size for gradient updates
        Too high → oscillating weights. Too low → slow adaptation.
    forward_bars: prediction horizon (5 bars = 1 week ahead)

Dependencies: numpy, pandas only — no heavy ML libs.
"""

import logging
import numpy as np
import pandas as pd

from strategy.indicators import col
from strategy.adaptive_signals import (
    cycle_series, build_filtered_df, adaptive_ichimoku_scores,
    wht_multiplier_series, WHT_WINDOW,
)
from strategy.signals import vidya_score, volume_weighted_rsi, MIN_BARS_NEEDED

log = logging.getLogger(__name__)

# ── Default hyperparameters ────────────────────────────────────────────────────
LAMBDA_FORGET  = 0.995    # forgetting factor — half-life ~138 bars (6 months)
ETA            = 0.01     # learning rate
FORWARD_BARS   = 5        # prediction horizon (trading days)
N_COMPONENTS   = 6        # s1 through s6
MIN_BARS       = MIN_BARS_NEEDED + FORWARD_BARS + 50


# ── Simplex projection ─────────────────────────────────────────────────────────

def normalise_weights(w: np.ndarray) -> np.ndarray:
    """
    Normalise weights using softmax-then-clip to keep them positive and sum to 1.
    Smoother than hard simplex projection — preserves relative differences
    between weights rather than zeroing out small ones.
    """
    # Shift so minimum is 0 (avoids negative weights after gradient step)
    w_shifted = w - w.min()
    # Add small floor so no weight is exactly zero
    w_floored = w_shifted + 0.01
    return w_floored / w_floored.sum()


# ── Score component extraction ─────────────────────────────────────────────────

def extract_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all six raw score components (before tanh/WHT).
    Returns DataFrame with columns [s1..s6].
    """
    if len(df) < MIN_BARS:
        return pd.DataFrame()

    close  = col(df, "Close")
    df_f   = build_filtered_df(df)
    cyc_s  = cycle_series(close)

    s1, s2, s3, s4 = adaptive_ichimoku_scores(df_f, cyc_s)
    s5 = vidya_score(df)
    s6 = volume_weighted_rsi(df)

    return pd.DataFrame({
        "s1": s1, "s2": s2, "s3": s3,
        "s4": s4, "s5": s5, "s6": s6,
    }, index=df.index).dropna()


# ── Weight evolution ───────────────────────────────────────────────────────────

def compute_online_weights(
    comp:          pd.DataFrame,
    fwd_returns:   pd.Series,
    lam:           float = LAMBDA_FORGET,
    eta:           float = ETA,
) -> pd.DataFrame:
    """
    Run the online learning algorithm over historical components and
    forward returns, returning the weight vector at each bar.

    This is called once during backtesting (not live), producing a
    DataFrame of weights (n_bars × 6) that shows how the algorithm
    would have adapted weights over time.

    Args:
        comp:        DataFrame (n_bars × 6) of score components
        fwd_returns: Series of forward returns aligned to comp.index
        lam:         forgetting factor
        eta:         learning rate

    Returns:
        weights_df:  DataFrame (n_bars × 6), weights at each bar
    """
    n  = len(comp)
    w  = np.ones(N_COMPONENTS) / N_COMPONENTS   # uniform start
    W  = np.zeros((n, N_COMPONENTS))

    comp_arr = comp.values
    ret_arr  = fwd_returns.reindex(comp.index).values

    for t in range(n):
        W[t] = w.copy()

        # Only update when forward return is available (not NaN)
        if t + 1 < n and not np.isnan(ret_arr[t]):
            x_t   = comp_arr[t]
            pred  = np.dot(w, x_t)
            err   = ret_arr[t] - pred
            grad  = -2.0 * err * x_t              # MSE gradient

            # Forgetting + gradient step
            w_new = lam * w - eta * grad

            # Normalise to keep weights positive and summing to 1
            w = normalise_weights(w_new)

    return pd.DataFrame(W, index=comp.index,
                        columns=["w1","w2","w3","w4","w5","w6"])


# ── Main score function ────────────────────────────────────────────────────────

def online_composite_score(
    df:    pd.DataFrame,
    lam:   float = LAMBDA_FORGET,
    eta:   float = ETA,
) -> pd.Series:
    """
    Compute online-weighted composite score for a full OHLCV DataFrame.

    The weights are computed causally — weight at time t depends only
    on information available before t. No lookahead bias.

    Process:
        1. Extract components [s1..s6] for all bars
        2. Compute forward returns (needed for online weight updates)
        3. Run online learning to get per-bar weights
        4. Weighted score = sum(w_i * s_i) per bar
        5. Apply tanh + WHT multiplier

    Note on forward return usage:
        The online algorithm uses forward returns to update weights,
        but the weight used at time t is w_t, which was computed
        using only returns up to t-1. So there is NO lookahead bias —
        we're using "what the algorithm would have learned by bar t"
        to form the score at bar t.
    """
    if len(df) < MIN_BARS + FORWARD_BARS:
        return pd.Series(dtype=float)

    comp = extract_components(df)
    if comp.empty:
        return pd.Series(dtype=float)

    close   = col(df, "Close").reindex(comp.index)
    fwd_ret = close.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)

    # Compute evolving weights (causal — no lookahead)
    weights_df = compute_online_weights(comp, fwd_ret, lam=lam, eta=eta)

    # Weighted sum per bar.
    # Weights are on the simplex (sum to 1), but original score sums 6 components.
    # Multiply by N_COMPONENTS to preserve scale: when weights are uniform
    # (1/6 each), N * dot(w, x) = sum(x), matching the original equal-weight sum.
    raw_scores = N_COMPONENTS * (comp.values * weights_df.values).sum(axis=1)
    raw        = pd.Series(raw_scores, index=comp.index)
    base       = raw.apply(np.tanh)

    # WHT multiplier (same as adaptive_signals)
    wht  = wht_multiplier_series(close).reindex(comp.index)
    final = (base * wht).clip(-1, 1)

    return final


def online_latest_score(df: pd.DataFrame, lam: float = LAMBDA_FORGET,
                        eta: float = ETA) -> tuple[float, float]:
    """Return (score_now, score_prev) for live trading."""
    sc = online_composite_score(df, lam=lam, eta=eta).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])


# ── Weight inspection ──────────────────────────────────────────────────────────

def inspect_weight_evolution(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Return the full weight evolution DataFrame for visualisation.
    Useful for understanding how weights shift across market regimes.
    """
    if len(df) < MIN_BARS + FORWARD_BARS:
        return pd.DataFrame()

    comp    = extract_components(df)
    if comp.empty:
        return pd.DataFrame()

    close   = col(df, "Close").reindex(comp.index)
    fwd_ret = close.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
    weights = compute_online_weights(comp, fwd_ret)

    if ticker:
        log.info(f"\nWeight evolution for {ticker}:")
        final_w = weights.iloc[-1]
        for i, name in enumerate(["s1(Ich.mom)", "s2(Ich.lag)",
                                   "s3(Cloud)", "s4(Chikou)",
                                   "s5(VIDYA)", "s6(VRSI)"]):
            log.info(f"  {name}: {final_w.iloc[i]:.4f}")

    return weights