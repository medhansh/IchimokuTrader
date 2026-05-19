"""
strategy/tiered_ma_signals.py
─────────────────────────────
Option 2: Tiered MA threshold filter.

Instead of a hard binary MA100/200 gate, the entry threshold scales with
how far the stock is from its MA100:

    effective_threshold = BASE_THRESH + penalty(ma_gap)

where:
    ma_gap  = (close - ma100) / ma100   (negative = below MA)
    penalty = max(0, -ma_gap) * SENSITIVITY

Examples (BASE_THRESH=0.25, SENSITIVITY=1.5):
    At MA100 (ma_gap=0):       threshold = 0.25  (unchanged)
    1% below MA100:             threshold = 0.265
    5% below MA100:             threshold = 0.325
    10% below MA100:            threshold = 0.40
    15% below MA100:            threshold = 0.475
    20% below MA100:            threshold = 0.55  (very selective)

Hard cutoffs still enforced:
    - Score never considered if stock >30% below MA100  (genuine falling knife)
    - If MA100 < MA200 * 0.95, add 0.05 extra penalty   (structural downtrend)

Drop-in replacement for the MA100/200 hard filter used in sheets_trader.py.
Exposes two public functions:

    tiered_ma_passes(close, ma100, ma200, score, base_thresh, sensitivity) -> bool
    compute_effective_threshold(close, ma100, ma200, base_thresh, sensitivity) -> float

And the batch signal-generation entry point:

    tiered_latest_score_and_threshold(df, base_thresh, sensitivity) -> (score, score_prev, eff_thresh)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── defaults ────────────────────────────────────────────────────────────────
BASE_THRESH    = 0.25   # same as adaptive baseline
SENSITIVITY    = 1.5    # threshold penalty per unit of ma_gap below 0
HARD_CUTOFF    = 0.30   # max allowed distance below MA100 (30%)
DOWNTREND_PEN  = 0.05   # extra penalty when MA100 < MA200*0.95
MAX_THRESH     = 0.60   # ceiling on effective threshold


# ─────────────────────────────────────────────────────────────────────────────
# Core threshold math
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_threshold(
    close: float,
    ma100: float,
    ma200: float,
    base_thresh: float = BASE_THRESH,
    sensitivity: float = SENSITIVITY,
) -> float:
    """
    Return the effective entry threshold for a single bar.

    Returns float('inf') when the stock is beyond the hard cutoff
    (i.e. the hard gate still applies for extreme cases).
    """
    if ma100 <= 0 or ma200 <= 0 or np.isnan(ma100) or np.isnan(ma200):
        return float("inf")

    ma_gap = (close - ma100) / ma100   # negative when below MA100

    # Hard cutoff: more than HARD_CUTOFF below MA100 → always block
    if ma_gap < -HARD_CUTOFF:
        return float("inf")

    # Base penalty from distance below MA100 (zero when above MA100)
    penalty = max(0.0, -ma_gap) * sensitivity

    # Additional penalty if MA100 itself is in a downtrend vs MA200
    if ma100 < ma200 * 0.95:
        penalty += DOWNTREND_PEN

    eff = base_thresh + penalty
    return min(eff, MAX_THRESH)


def tiered_ma_passes(
    close: float,
    ma100: float,
    ma200: float,
    score: float,
    base_thresh: float = BASE_THRESH,
    sensitivity: float = SENSITIVITY,
) -> bool:
    """Return True if score clears the tiered effective threshold."""
    eff = compute_effective_threshold(close, ma100, ma200, base_thresh, sensitivity)
    return score >= eff


# ─────────────────────────────────────────────────────────────────────────────
# Re-use adaptive score computation from adaptive_signals
# ─────────────────────────────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Extract a Series from a DataFrame column safely."""
    col = df[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.squeeze()


def tiered_latest_score_and_threshold(
    df: pd.DataFrame,
    base_thresh: float = BASE_THRESH,
    sensitivity: float = SENSITIVITY,
    ma100_period: int = 100,
    ma200_period: int = 200,
) -> tuple[float, float, float]:
    """
    Given a price DataFrame (OHLCV), compute:
        (latest_score, prev_score, effective_threshold)

    Imports adaptive_latest_score for the signal; adds the tiered MA gate on top.
    Returns (score, score_prev, eff_thresh).  The caller should check
        score_prev < eff_thresh <= score    ← fresh crossover above threshold
        score_prev > -eff_thresh >= score   ← fresh crossover below threshold
    """
    # Lazy import to avoid circular deps
    from strategy.adaptive_signals import adaptive_latest_score

    score, score_prev = adaptive_latest_score(df)

    close  = _safe_col(df, "Close").iloc[-1]
    ma100  = _safe_col(df, "Close").rolling(ma100_period).mean().iloc[-1]
    ma200  = _safe_col(df, "Close").rolling(ma200_period).mean().iloc[-1]

    eff_thresh = compute_effective_threshold(close, ma100, ma200, base_thresh, sensitivity)

    return score, score_prev, eff_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised batch version (used in the backtester)
# ─────────────────────────────────────────────────────────────────────────────

def compute_tiered_signals(
    close: pd.Series,
    score: pd.Series,
    base_thresh: float = BASE_THRESH,
    sensitivity: float = SENSITIVITY,
    ma100_period: int = 100,
    ma200_period: int = 200,
) -> pd.DataFrame:
    """
    Compute effective threshold and entry/exit signals for a single ticker.

    Parameters
    ----------
    close   : daily close prices
    score   : composite signal score (from adaptive_signals)
    base_thresh, sensitivity : tiered MA params

    Returns
    -------
    DataFrame with columns:
        ma100, ma200, ma_gap, eff_thresh, entry, exit
    """
    ma100 = close.rolling(ma100_period).mean()
    ma200 = close.rolling(ma200_period).mean()
    ma_gap = (close - ma100) / ma100

    # Penalty
    penalty = ma_gap.clip(upper=0).abs() * sensitivity

    # Extra structural downtrend penalty
    downtrend_mask = ma100 < ma200 * 0.95
    penalty = penalty + downtrend_mask.astype(float) * DOWNTREND_PEN

    # Hard cutoff → set penalty to inf
    hard_block = ma_gap < -HARD_CUTOFF
    penalty = penalty.where(~hard_block, np.inf)

    eff_thresh = (base_thresh + penalty).clip(upper=MAX_THRESH)
    eff_thresh = eff_thresh.where(~hard_block, np.inf)

    # Entry: score crosses above eff_thresh
    above      = score >= eff_thresh
    entry      = above & ~above.shift(1).astype("boolean").fillna(False).astype(bool)

    # Exit: score crosses below -eff_thresh  (symmetric exit)
    below      = score <= -eff_thresh
    exit_sig   = below & ~below.shift(1).astype("boolean").fillna(False).astype(bool)

    return pd.DataFrame({
        "ma100":      ma100,
        "ma200":      ma200,
        "ma_gap":     ma_gap,
        "eff_thresh": eff_thresh,
        "entry":      entry,
        "exit":       exit_sig,
    }, index=close.index)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics helper (mirrors _write_signal_radar logic in sheets_trader)
# ─────────────────────────────────────────────────────────────────────────────

def signal_radar_row(
    ticker: str,
    df: pd.DataFrame,
    score: float,
    score_prev: float,
    base_thresh: float = BASE_THRESH,
    sensitivity: float = SENSITIVITY,
) -> dict:
    """
    Return a diagnostic dict for one ticker — used in the backtest comparison
    to understand why a stock was or wasn't selected.
    """
    close = _safe_col(df, "Close").iloc[-1]
    ma100 = _safe_col(df, "Close").rolling(100).mean().iloc[-1]
    ma200 = _safe_col(df, "Close").rolling(200).mean().iloc[-1]
    ma_gap = (close - ma100) / ma100

    eff = compute_effective_threshold(close, ma100, ma200, base_thresh, sensitivity)
    passes = score >= eff

    if eff == float("inf"):
        reason = f"Hard block (ma_gap={ma_gap:.1%} < -{HARD_CUTOFF:.0%})"
    elif score >= eff:
        reason = "PASS"
    elif score_prev >= eff:
        reason = "Already above (no crossover)"
    else:
        reason = f"Score too low ({score:.3f} vs eff_thresh {eff:.3f})"

    return {
        "ticker":     ticker,
        "score":      round(score, 4),
        "score_prev": round(score_prev, 4),
        "close":      round(close, 2),
        "ma100":      round(ma100, 2),
        "ma200":      round(ma200, 2),
        "ma_gap_pct": round(ma_gap * 100, 1),
        "eff_thresh": round(eff, 3) if eff != float("inf") else "∞",
        "passes":     passes,
        "reason":     reason,
    }