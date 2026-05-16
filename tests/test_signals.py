"""
tests/test_signals.py
Run: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from strategy.indicators import ichimoku, atr, adx, volume_weighted_rsi
from strategy.signals    import composite_score, entry_signal, exit_signal, size_position


def _fake_ohlcv(n=500, drift=0.0002, vol=0.015, seed=42):
    """Generate synthetic OHLCV for testing."""
    rng     = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    close   = 1000 * np.exp(np.cumsum(returns))
    high    = close * (1 + rng.uniform(0.002, 0.015, n))
    low     = close * (1 - rng.uniform(0.002, 0.015, n))
    volume  = rng.lognormal(np.log(1_000_000), 0.5, n)
    idx     = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


DF = _fake_ohlcv()


class TestIndicators:
    def test_ichimoku_returns_all_components(self):
        ic = ichimoku(DF)
        assert set(ic.keys()) == {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou"}
        for v in ic.values():
            assert isinstance(v, pd.Series)
            assert len(v) == len(DF)

    def test_atr_positive(self):
        a = atr(DF)
        assert (a.dropna() > 0).all()

    def test_adx_columns(self):
        d = adx(DF)
        assert {"adx", "di_plus", "di_minus"} <= set(d.columns)
        assert (d["adx"].dropna() >= 0).all()

    def test_vrsi_bounded(self):
        s = volume_weighted_rsi(DF)
        valid = s.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()


class TestSignals:
    def test_composite_score_range(self):
        sc = composite_score(DF).dropna()
        assert not sc.empty
        assert (sc >= -1).all() and (sc <= 1).all()

    def test_insufficient_data_returns_empty(self):
        tiny = DF.head(50)
        sc = composite_score(tiny)
        assert sc.empty

    def test_entry_signal(self):
        # Crosses from below
        assert entry_signal(0.45, 0.38) is True
        # Already above — not a crossover
        assert entry_signal(0.50, 0.45) is False
        # Below threshold
        assert entry_signal(0.30, 0.25) is False

    def test_exit_signal(self):
        assert exit_signal(-0.45, -0.35) is True
        assert exit_signal(-0.30, -0.25) is False

    def test_size_position_returns_int(self):
        sh = size_position(close=500.0, atr_val=10.0, score=0.6,
                           total_capital=200_000)
        assert isinstance(sh, int)
        assert sh >= 1

    def test_size_position_zero_atr(self):
        sh = size_position(close=500.0, atr_val=0.0, score=0.6,
                           total_capital=200_000)
        assert sh == 0

    def test_size_position_cap(self):
        # Should never exceed 10% of capital
        sh = size_position(close=1.0, atr_val=0.001, score=1.0,
                           total_capital=200_000, max_pos_pct=0.10)
        assert sh * 1.0 <= 200_000 * 0.10 + 1   # +1 for floor rounding