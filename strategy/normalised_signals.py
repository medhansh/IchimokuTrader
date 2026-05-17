"""
strategy/normalised_signals.py
--------------------------------
Two improved variants of the adaptive strategy:

    Model A: Adaptive + Direction 1  (rolling z-score normalisation)
    ─────────────────────────────────────────────────────────────────
    Each score component s1-s6 is z-scored against its own rolling
    252-bar distribution before summing. This makes each component
    stationary and comparable across regimes — a component that is
    "loud" in a trending market doesn't drown out components that are
    "quiet" in that regime. The equal-weight sum then becomes genuinely
    regime-agnostic rather than accidentally dominated by whichever
    component happens to have the largest variance in the current period.

    Model B: Adaptive + Direction 1 + Direction 3  (adaptive threshold)
    ──────────────────────────────────────────────────────────────────────
    Extends Model A with a self-calibrating confidence multiplier based
    on recent signal accuracy:

        accuracy_t = exponential moving average of (signal_correct ? 1 : 0)
        confidence = rescaled accuracy to [CONF_MIN, CONF_MAX]
        final_score = normalised_score * confidence

    When the strategy has been misfiring recently (wrong direction calls),
    confidence drops and the effective threshold rises — fewer trades taken.
    When it's been accurate, confidence rises — more trades taken.

    This requires no regime label, no model training, no lookahead.
    It simply asks: "has this strategy been right recently?" and acts
    accordingly. The exponential forgetting ensures it adapts within weeks.

Both models use the same underlying components as adaptive_composite_score
(FFT cycle detection, filtered price, adaptive Ichimoku, VIDYA, VRSI, WHT).
"""

import numpy as np
import pandas as pd

from strategy.indicators import col
from strategy.adaptive_signals import (
    build_filtered_df, cycle_series, adaptive_ichimoku_scores,
    wht_multiplier_series, vidya_score, volume_weighted_rsi,
    FFT_WINDOW, WHT_WINDOW, CYCLE_MAX, SENKOB_RATIO,
)

# ── Configuration ──────────────────────────────────────────────────────────────

ZSCORE_WINDOW  = 252     # rolling window for z-score normalisation (1 trading year)
ZSCORE_MIN_OBS = 63      # minimum observations before z-score is applied (1 quarter)

CONF_WINDOW    = 60      # exponential window for accuracy tracking (~3 months)
CONF_ALPHA     = 2 / (CONF_WINDOW + 1)   # EMA alpha
CONF_MIN       = 0.5     # minimum confidence multiplier (strategy badly misfiring)
CONF_MAX       = 1.3     # maximum confidence multiplier (strategy consistently right)
FORWARD_BARS   = 5       # bars ahead to measure signal accuracy

MIN_BARS = FFT_WINDOW + WHT_WINDOW + int(CYCLE_MAX * SENKOB_RATIO) + ZSCORE_WINDOW + 10


# ── Rolling z-score normalisation (Direction 1) ────────────────────────────────

def zscore_normalise(s: pd.Series, window: int = ZSCORE_WINDOW,
                     min_obs: int = ZSCORE_MIN_OBS) -> pd.Series:
    """
    Mean-and-std normalisation (original Dir1).
    Kept for reference but NOT used in the active strategies.
    Removes both variance AND trend — wrong for momentum strategies.
    """
    roll_mean = s.rolling(window, min_periods=min_obs).mean()
    roll_std  = s.rolling(window, min_periods=min_obs).std().replace(0, np.nan)
    z = (s - roll_mean) / roll_std
    z = z.where(z.notna(), s)
    return z.clip(-3, 3) / 3.0


def stddev_normalise(s: pd.Series, window: int = ZSCORE_WINDOW,
                     min_obs: int = ZSCORE_MIN_OBS) -> pd.Series:
    """
    Std-only normalisation — correct for momentum strategies.

    Divides each component by its rolling standard deviation WITHOUT
    subtracting the mean. This removes heteroscedasticity (components
    becoming artificially loud in high-volatility regimes) while
    preserving the directional trend signal.

    A component that has been persistently positive (e.g. Ichimoku
    momentum during a bull run) stays positive after normalisation.
    Only its magnitude relative to recent volatility is adjusted.

        z_t = s_t / σ_{t-window:t}

    Clipped to [-3, 3] then rescaled to [-1, 1] to match the output
    range of the raw components for downstream tanh compatibility.
    """
    roll_std = s.rolling(window, min_periods=min_obs).std().replace(0, np.nan)
    z = s / roll_std
    z = z.where(z.notna(), s)   # fall back to raw where insufficient history
    return z.clip(-3, 3) / 3.0


def _extract_raw_components(df: pd.DataFrame):
    """
    Extract all six raw score components (before tanh/WHT/normalisation).
    Returns (s1, s2, s3, s4, s5, s6) as pandas Series.
    """
    close = col(df, "Close")
    df_f  = build_filtered_df(df)
    cyc_s = cycle_series(close)
    s1, s2, s3, s4 = adaptive_ichimoku_scores(df_f, cyc_s)
    s5 = vidya_score(df)
    s6 = volume_weighted_rsi(df)
    return s1, s2, s3, s4, s5, s6


# ── Model A: Adaptive + Dir1+3 (mean+std z-score + confidence) ────────────────
# Kept for reference. Removes trend via mean subtraction — wrong for momentum.

def normalised_composite_score(df: pd.DataFrame) -> pd.Series:
    """Full z-score (mean+std) — reference only, not recommended for momentum."""
    if len(df) < MIN_BARS:
        return pd.Series(dtype=float)
    s1, s2, s3, s4, s5, s6 = _extract_raw_components(df)
    close = col(df, "Close")
    raw   = (zscore_normalise(s1) + zscore_normalise(s2) +
             zscore_normalise(s3) + zscore_normalise(s4) +
             zscore_normalise(s5) + zscore_normalise(s6))
    return (raw.apply(np.tanh) * wht_multiplier_series(close)).clip(-1, 1)


def _compute_confidence(
    score_series: pd.Series,
    close:        pd.Series,
    threshold:    float = 0.25,
) -> pd.Series:
    """
    Causal confidence multiplier from recent signal accuracy.

    Tracks a per-bar EMA of whether recent entry signals (score crossing
    threshold) were followed by returns in the predicted direction.

    Confidence ∈ [CONF_MIN, CONF_MAX]:
        accuracy → 1.0  →  CONF_MAX  (strategy working — amplify)
        accuracy → 0.0  →  CONF_MIN  (strategy misfiring — dampen)
        no recent signals → 1.0       (neutral)

    Fully causal: outcome at bar t uses close[t+FORWARD_BARS] which is
    only available FORWARD_BARS bars later. No lookahead.
    """
    n       = len(score_series)
    acc_ema = np.full(n, 0.5)
    conf    = np.ones(n)
    prices  = close.reindex(score_series.index).values
    scores  = score_series.values

    for t in range(1, n):
        prev_s       = scores[t-1]
        curr_s       = scores[t]
        signal_fired = (curr_s >= threshold and prev_s < threshold)

        if signal_fired and t + FORWARD_BARS < n:
            if prices[t] > 0 and prices[t + FORWARD_BARS] > 0:
                fwd_ret    = prices[t + FORWARD_BARS] / prices[t] - 1
                correct    = 1.0 if (curr_s > 0) == (fwd_ret > 0) else 0.0
                acc_ema[t] = CONF_ALPHA * correct + (1 - CONF_ALPHA) * acc_ema[t-1]
            else:
                acc_ema[t] = acc_ema[t-1]
        else:
            acc_ema[t] = acc_ema[t-1]

        conf[t] = CONF_MIN + acc_ema[t] * (CONF_MAX - CONF_MIN)

    return pd.Series(conf, index=score_series.index)


def normalised_adaptive_composite_score(df: pd.DataFrame,
                                        threshold: float = 0.25) -> pd.Series:
    """Adaptive + full z-score + confidence (current Dir1+3, mean removed)."""
    if len(df) < MIN_BARS + FORWARD_BARS:
        return pd.Series(dtype=float)
    base = normalised_composite_score(df)
    if base.dropna().empty:
        return pd.Series(dtype=float)
    close = col(df, "Close")
    conf  = _compute_confidence(base, close.reindex(base.index), threshold)
    return (base * conf).clip(-1, 1)


def normalised_adaptive_latest_score(df: pd.DataFrame,
                                     threshold: float = 0.25) -> tuple[float, float]:
    sc = normalised_adaptive_composite_score(df, threshold).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])


# ── Model B: Adaptive + Std-only Dir1+3 (std-only + confidence) ───────────────

def stddev_normalised_composite_score(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive + std-only normalisation.

    Divides each component by its rolling std WITHOUT subtracting mean.
    Removes heteroscedasticity (variance inflation in high-vol regimes)
    while preserving trend direction — correct for momentum strategies.
    """
    if len(df) < MIN_BARS:
        return pd.Series(dtype=float)
    s1, s2, s3, s4, s5, s6 = _extract_raw_components(df)
    close = col(df, "Close")
    raw   = (stddev_normalise(s1) + stddev_normalise(s2) +
             stddev_normalise(s3) + stddev_normalise(s4) +
             stddev_normalise(s5) + stddev_normalise(s6))
    return (raw.apply(np.tanh) * wht_multiplier_series(close)).clip(-1, 1)


def stddev_normalised_adaptive_composite_score(df: pd.DataFrame,
                                               threshold: float = 0.25) -> pd.Series:
    """
    Adaptive + std-only normalisation + adaptive confidence.

    The primary hypothesis:
        1. Std-only normalisation keeps trend signal, removes variance noise
        2. Confidence adapts exposure to recent signal accuracy

    Expected behaviour vs baseline:
        Bull market:  similar returns (trend preserved), better Calmar
        Bear market:  fewer false signals (high vol periods damped)
        Random noise: confidence learns misfiring and reduces exposure
    """
    if len(df) < MIN_BARS + FORWARD_BARS:
        return pd.Series(dtype=float)
    base = stddev_normalised_composite_score(df)
    if base.dropna().empty:
        return pd.Series(dtype=float)
    close = col(df, "Close")
    conf  = _compute_confidence(base, close.reindex(base.index), threshold)
    return (base * conf).clip(-1, 1)


def stddev_adaptive_latest_score(df: pd.DataFrame,
                                 threshold: float = 0.25) -> tuple[float, float]:
    """Return (score_now, score_prev) for live trading."""
    sc = stddev_normalised_adaptive_composite_score(df, threshold).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])