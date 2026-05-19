"""
tests/test_tiered_ma.py
────────────────────────
Unit tests for strategy/tiered_ma_signals.py
Run: pytest tests/test_tiered_ma.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.tiered_ma_signals import (
    compute_effective_threshold,
    tiered_ma_passes,
    compute_tiered_signals,
    BASE_THRESH, SENSITIVITY, HARD_CUTOFF, DOWNTREND_PEN, MAX_THRESH,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_effective_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEffectiveThreshold:

    def test_at_ma100_returns_base(self):
        t = compute_effective_threshold(close=100, ma100=100, ma200=90)
        assert math.isclose(t, BASE_THRESH, abs_tol=1e-9)

    def test_above_ma100_returns_base(self):
        t = compute_effective_threshold(close=110, ma100=100, ma200=90)
        assert math.isclose(t, BASE_THRESH, abs_tol=1e-9)

    def test_5pct_below_ma100(self):
        t = compute_effective_threshold(close=95, ma100=100, ma200=90)
        expected = BASE_THRESH + 0.05 * SENSITIVITY
        assert math.isclose(t, expected, abs_tol=1e-9)

    def test_10pct_below_ma100(self):
        t = compute_effective_threshold(close=90, ma100=100, ma200=90)
        expected = BASE_THRESH + 0.10 * SENSITIVITY
        assert math.isclose(t, expected, abs_tol=1e-9)

    def test_hard_cutoff_returns_inf(self):
        close = 100 * (1 - HARD_CUTOFF - 0.01)
        t = compute_effective_threshold(close=close, ma100=100, ma200=80)
        assert t == math.inf

    def test_exactly_at_hard_cutoff_not_blocked(self):
        close = 100 * (1 - HARD_CUTOFF)
        t = compute_effective_threshold(close=close, ma100=100, ma200=80)
        assert t != math.inf

    def test_downtrend_penalty_applied(self):
        close = 100; ma100 = 100; ma200 = 110
        t = compute_effective_threshold(close, ma100, ma200)
        expected = BASE_THRESH + DOWNTREND_PEN
        assert math.isclose(t, expected, abs_tol=1e-9)

    def test_downtrend_penalty_not_applied_when_close(self):
        close = 100; ma100 = 100; ma200 = 104
        t = compute_effective_threshold(close, ma100, ma200)
        assert math.isclose(t, BASE_THRESH, abs_tol=1e-9)

    def test_capped_at_max_thresh(self):
        close = 100 * (1 - HARD_CUTOFF + 0.01)
        t = compute_effective_threshold(close=close, ma100=100, ma200=90)
        assert t <= MAX_THRESH

    def test_nan_ma_returns_inf(self):
        t = compute_effective_threshold(close=100, ma100=float("nan"), ma200=90)
        assert t == math.inf

    def test_zero_ma_returns_inf(self):
        t = compute_effective_threshold(close=100, ma100=0, ma200=90)
        assert t == math.inf


# ─────────────────────────────────────────────────────────────────────────────
# tiered_ma_passes  (scalar — definitive hard-cutoff test lives here)
# ─────────────────────────────────────────────────────────────────────────────

class TestTieredMaPasses:

    def test_passes_when_score_above_effective_threshold(self):
        assert tiered_ma_passes(100, 100, 90, score=0.30) is True

    def test_fails_when_score_below_effective_threshold(self):
        assert tiered_ma_passes(100, 100, 90, score=0.20) is False

    def test_fails_when_hard_blocked(self):
        # 35% below MA100 → beyond HARD_CUTOFF → must block even score=0.99
        close = 100 * (1 - HARD_CUTOFF - 0.05)
        assert tiered_ma_passes(close, 100, 80, score=0.99) is False

    def test_passes_with_strong_score_below_ma(self):
        # 5% below MA100 → threshold = 0.325; score 0.35 should pass
        assert tiered_ma_passes(95, 100, 90, score=0.35) is True

    def test_fails_with_weak_score_below_ma(self):
        # 5% below MA100 → threshold = 0.325; score 0.26 should fail
        assert tiered_ma_passes(95, 100, 90, score=0.26) is False


# ─────────────────────────────────────────────────────────────────────────────
# compute_tiered_signals (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTieredSignals:

    def _make_series(self, values, start="2020-01-01"):
        idx = pd.date_range(start, periods=len(values), freq="B")
        return pd.Series(values, index=idx)

    def test_returns_expected_columns(self):
        n = 250
        close = self._make_series([100.0] * n)
        score = self._make_series([0.0]   * n)
        result = compute_tiered_signals(close, score)
        for col in ("ma100", "ma200", "eff_thresh", "entry", "exit"):
            assert col in result.columns

    def test_entry_fires_on_crossover_above_ma(self):
        """Score crosses 0.25 threshold while price is above MA100."""
        n = 250
        prices = [100.0] * n
        scores = [0.20] * 204 + [0.30] * (n - 204)
        close = self._make_series(prices)
        score = self._make_series(scores)
        result = compute_tiered_signals(close, score)
        entries = result["entry"]
        assert bool(entries.iloc[204]) is True
        assert bool(entries.iloc[203]) is False
        assert bool(entries.iloc[205]) is False

    def test_no_entry_in_sustained_20pct_drop_zone(self):
        """
        500 bars stable at 100, then 80 bars where price drops to 80 (20% below).

        MA100 lags — after 80 bars the MA100 is still ~84, so ma_gap ≈ -5%..−20%.
        At any point in those 80 bars, eff_thresh > BASE_THRESH + 0.01 (score),
        so a score of 0.26 should NOT produce an entry crossover in the first 80 bars.

        We deliberately test only the first 80 bars of the drop (before MA100
        converges to the new price), which is the regime we care about: a stock
        that has *recently* broken below its MA100 and should require a stronger
        signal to enter.
        """
        n_stable = 500
        n_drop   = 80      # short enough that MA100 hasn't converged to 80 yet
        prices   = [100.0] * n_stable + [80.0] * n_drop
        # Score is just a hair above BASE_THRESH — passes old hard filter, fails tiered
        scores   = [0.0] * n_stable + [BASE_THRESH + 0.01] * n_drop
        close    = self._make_series(prices)
        score    = self._make_series(scores)
        result   = compute_tiered_signals(close, score)

        # In the first n_drop bars after the break, eff_thresh > 0.26 because
        # MA100 is still well above 80, making ma_gap meaningfully negative.
        drop_entries = result["entry"].iloc[n_stable : n_stable + n_drop]
        assert drop_entries.sum() == 0, (
            f"Expected 0 entries in first {n_drop} drop bars, got {drop_entries.sum()}.\n"
            f"eff_thresh range: {result['eff_thresh'].iloc[n_stable:n_stable+n_drop].min():.3f}"
            f" – {result['eff_thresh'].iloc[n_stable:n_stable+n_drop].max():.3f}"
        )

    def test_entry_fires_with_strong_score_in_drop_zone(self):
        """
        Same 20%-below-MA100 scenario but with a very strong score (0.60).
        0.60 > 0.55 → should eventually fire a crossover entry.
        """
        n_stable = 500
        n_drop   = 200
        prices   = [100.0] * n_stable + [80.0] * n_drop
        # Score below threshold during stable, then strong crossover in drop zone
        scores   = [0.0] * n_stable + [0.60] * n_drop
        close    = self._make_series(prices)
        score    = self._make_series(scores)
        result   = compute_tiered_signals(close, score)

        drop_entries = result["entry"].iloc[n_stable:]
        # Should see exactly 1 crossover (the first bar score goes above eff_thresh)
        assert drop_entries.sum() >= 1, "Expected at least one entry with strong score"

    def test_higher_threshold_when_below_ma(self):
        """Price 10% below MA100 → eff_thresh > BASE_THRESH."""
        n = 300
        prices = [100.0] * 200 + [90.0] * 100
        close  = self._make_series(prices)
        score  = self._make_series([0.0] * n)
        result = compute_tiered_signals(close, score)
        eff_in_zone = result["eff_thresh"].iloc[250:].dropna()
        assert (eff_in_zone > BASE_THRESH).all()

    def test_eff_thresh_equals_base_when_above_ma(self):
        """Price above MA100 → eff_thresh = BASE_THRESH."""
        n     = 300
        close = self._make_series([110.0] * n)
        score = self._make_series([0.0]   * n)
        result = compute_tiered_signals(close, score)
        eff = result["eff_thresh"].iloc[210:].dropna()
        assert (np.abs(eff - BASE_THRESH) < 1e-9).all()