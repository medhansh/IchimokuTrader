"""
strategy/signals.py
--------------------
Combines indicators into a composite score and generates entry/exit signals.

Dependencies: numpy, pandas, strategy.indicators only.
"""

import numpy as np
import pandas as pd

from strategy.indicators import ichimoku_scores, vidya_score, volume_weighted_rsi


# ── Parameters ─────────────────────────────────────────────────────────────────

TENKAN_PERIOD   = 32
KIJUN_PERIOD    = 96
SENKOB_PERIOD   = 192
DISPLACEMENT    = 96
SCORE_BUY       = 0.40
SCORE_SELL      = -0.40
MIN_BARS_NEEDED = DISPLACEMENT + SENKOB_PERIOD + 50   # ~338 bars minimum


# ── Core score ─────────────────────────────────────────────────────────────────

def composite_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Ichimoku + VIDYA + VRSI composite score.

    Score = tanh(s1 + s2 + s3 + s4 + s5 + s6)

    Where:
        s1, s2 = Ichimoku momentum (tenkan vs kijun, lagged)
        s3     = price position vs cloud
        s4     = chikou confirmation
        s5     = VIDYA trend (short vs long)
        s6     = volume-weighted RSI

    Output range: approximately [-1, 1]
    Values > SCORE_BUY  → bullish
    Values < SCORE_SELL → bearish
    """
    if len(df) < MIN_BARS_NEEDED:
        return pd.Series(dtype=float)

    s1, s2, s3, s4 = ichimoku_scores(
        df,
        tenkan_period=TENKAN_PERIOD,
        kijun_period=KIJUN_PERIOD,
        senkob_period=SENKOB_PERIOD,
        displacement=DISPLACEMENT,
    )
    s5 = vidya_score(df)
    s6 = volume_weighted_rsi(df)

    raw = s1 + s2 + s3 + s4 + s5 + s6
    return raw.apply(np.tanh)


def latest_score(df: pd.DataFrame) -> tuple[float, float]:
    """
    Return (score_now, score_prev) for the latest two bars.
    Returns (0.0, 0.0) if insufficient data.
    """
    sc = composite_score(df).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])


# ── Signal generation ──────────────────────────────────────────────────────────

def entry_signal(score_now: float, score_prev: float) -> bool:
    """True when score crosses above BUY threshold (crossover, not level)."""
    return score_now >= SCORE_BUY and score_prev < SCORE_BUY


def exit_signal(score_now: float, score_prev: float) -> bool:
    """True when score crosses below SELL threshold."""
    return score_now <= SCORE_SELL and score_prev > SCORE_SELL


# ── For backtesting (vectorbt-compatible) ──────────────────────────────────────

def signals_for_backtest(
    score_df: pd.DataFrame,
    buy_thresh:  float = SCORE_BUY,
    sell_thresh: float = SCORE_SELL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate entry/exit boolean DataFrames from a wide score DataFrame.
    score_df columns = tickers, index = dates.

    Used by backtest/vectorbt_runner.py only.
    """
    entries = (score_df >= buy_thresh)  & (score_df.shift(1) < buy_thresh)
    exits   = (score_df <= sell_thresh) & (score_df.shift(1) > sell_thresh)
    return entries.fillna(False), exits.fillna(False)


# ── Position sizing ────────────────────────────────────────────────────────────

def size_position(
    close:          float,
    atr_val:        float,
    score:          float,
    total_capital:  float,
    risk_pct:       float = 0.02,
    atr_multiplier: float = 2.5,
    max_pos_pct:    float = 0.10,
    threshold:      float = SCORE_BUY,
) -> int:
    """
    ATR risk sizing × score-proportional scaling.

    Step 1: base_shares = (capital × risk_pct) / (ATR × multiplier)
            → equal rupee risk across all positions
    Step 2: score_factor = (score - threshold) / (1 - threshold)
            → scales [0.1, 1.0] with conviction
    Step 3: cap at max_pos_pct of capital per position

    Returns whole number of shares (minimum 1).
    """
    import math
    if atr_val <= 0 or close <= 0:
        return 0

    risk_rupees   = total_capital * risk_pct
    stop_distance = atr_val * atr_multiplier
    base_shares   = risk_rupees / stop_distance

    score_factor  = max(0.1, min(1.0, (score - threshold) / (1.0 - threshold + 1e-9)))
    shares        = max(1, math.floor(base_shares * score_factor))

    max_shares    = max(1, math.floor(total_capital * max_pos_pct / close))
    return min(shares, max_shares)