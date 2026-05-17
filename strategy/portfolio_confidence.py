"""
strategy/portfolio_confidence.py
---------------------------------
Portfolio-level adaptive threshold.

Instead of modifying per-ticker scores (which has insufficient data per
ticker to learn from), this operates at the portfolio level — the right
abstraction for a multi-stock momentum strategy.

Mechanism:
    After each week, measure the portfolio's recent return. If the
    portfolio has been underperforming (recent return negative), raise
    the entry threshold — be more selective. If performing well, keep
    the threshold at the baseline.

    effective_threshold(t) = base_threshold + penalty(t)
    penalty(t) = max(0, -recent_portfolio_return(t)) * sensitivity

    Examples:
        Portfolio up 1% last week   → penalty=0    → threshold=0.25
        Portfolio flat last week    → penalty=0    → threshold=0.25
        Portfolio down 1% last week → penalty=0.02 → threshold=0.27
        Portfolio down 3% last week → penalty=0.06 → threshold=0.31

    The threshold is clamped to [base, max_threshold] so it never
    becomes so restrictive that no trades are taken.

This is applied in the backtester and live bot as a wrapper around
the existing adaptive_composite_score — the score itself is unchanged.
"""

import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────

BASE_THRESHOLD  = 0.25    # baseline entry threshold (from backtest optimisation)
MAX_THRESHOLD   = 0.45    # ceiling — never block all entries
SENSITIVITY     = 2.0     # how much threshold rises per unit of negative return
                          # e.g. sensitivity=2: down 2% → threshold += 0.04
LOOKBACK_BARS   = 5       # recent performance window (1 trading week)


# ── Portfolio-level adaptive threshold ────────────────────────────────────────

class PortfolioThreshold:
    """
    Tracks portfolio equity and computes adaptive entry threshold.

    Usage in backtester:
        pt = PortfolioThreshold()
        # Each bar:
        pt.update(current_portfolio_value)
        threshold = pt.current_threshold()

    Usage in live bot:
        pt = PortfolioThreshold.from_json(state_file)
        pt.update(portfolio_value_today)
        threshold = pt.current_threshold()
        pt.save(state_file)
    """

    def __init__(self,
                 base:        float = BASE_THRESHOLD,
                 max_thresh:  float = MAX_THRESHOLD,
                 sensitivity: float = SENSITIVITY,
                 lookback:    int   = LOOKBACK_BARS):
        self.base        = base
        self.max_thresh  = max_thresh
        self.sensitivity = sensitivity
        self.lookback    = lookback
        self._history: list[float] = []   # rolling portfolio values

    def update(self, portfolio_value: float):
        """Call once per bar with the current marked-to-market portfolio value."""
        self._history.append(float(portfolio_value))
        # Keep only what we need
        if len(self._history) > self.lookback + 1:
            self._history.pop(0)

    def recent_return(self) -> float:
        """Return over the last lookback bars. 0.0 if insufficient history."""
        if len(self._history) < self.lookback + 1:
            return 0.0
        oldest = self._history[0]
        newest = self._history[-1]
        return (newest / oldest - 1) if oldest > 0 else 0.0

    def current_threshold(self) -> float:
        """
        Compute effective entry threshold for the current bar.

        threshold = base + max(0, -recent_return) * sensitivity
        clamped to [base, max_threshold]
        """
        r       = self.recent_return()
        penalty = max(0.0, -r) * self.sensitivity
        return float(np.clip(self.base + penalty, self.base, self.max_thresh))

    def state_dict(self) -> dict:
        return {
            "history":     self._history,
            "base":        self.base,
            "max_thresh":  self.max_thresh,
            "sensitivity": self.sensitivity,
            "lookback":    self.lookback,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioThreshold":
        pt = cls(
            base        = d.get("base",        BASE_THRESHOLD),
            max_thresh  = d.get("max_thresh",  MAX_THRESHOLD),
            sensitivity = d.get("sensitivity", SENSITIVITY),
            lookback    = d.get("lookback",    LOOKBACK_BARS),
        )
        pt._history = d.get("history", [])
        return pt