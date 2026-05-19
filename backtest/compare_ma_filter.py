"""
backtest/compare_ma_filter.py
──────────────────────────────
3-way comparison on the same adaptive signal (adaptive_signals.py):

    Strategy A  — Hard MA filter (current sheets_trader behaviour)
                  Requires close > MA100 AND MA100 > MA200 * 0.97

    Strategy B  — Tiered MA filter (Option 2)
                  threshold = BASE + max(0, -ma_gap) * SENSITIVITY
                  Hard block only if stock > 30% below MA100

    Strategy C  — No MA filter (raw adaptive signal)
                  Pure score crossover at fixed 0.25 threshold

Run:
    python -m backtest.compare_ma_filter

Outputs:
    • Console table with Return / CAGR / Sharpe / MDD / Calmar / WinRate / Trades
    • PNG chart saved to data/backtest_results/ma_filter_comparison.png
    • CSV of all trades saved per strategy to data/backtest_results/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.data    import fetch_ohlcv
from bot.tickers import NIFTY500
from strategy.adaptive_signals  import adaptive_latest_score   # noqa: F401 (import check)
from strategy.tiered_ma_signals import (
    compute_effective_threshold,
    compute_tiered_signals,
    BASE_THRESH, SENSITIVITY,
)

RESULTS_DIR = ROOT / "data" / "backtest_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── backtest parameters ───────────────────────────────────────────────────────
TICKERS         = NIFTY500          # full universe
TRAIN_START     = "2019-01-01"      # data fetch start (warmup)
TEST_START      = "2021-01-01"      # backtest period start
TEST_END        = "2025-05-01"      # backtest period end
INIT_CAPITAL    = 2_00_000          # ₹2L
MAX_POSITIONS   = 8
ATR_PERIOD      = 14
ATR_MULT        = 2.5               # trailing stop distance
SCORE_THRESH    = 0.25              # base / no-filter threshold
SLIPPAGE        = 0.001             # 0.1% round-trip

# Hard MA filter params (Strategy A — matches sheets_trader.py)
HARD_MA100_MULT = 1.0               # close must be > ma100 * this
HARD_MA200_MULT = 0.97              # ma100 must be > ma200 * this

# Tiered MA filter params (Strategy B)
TIERED_BASE      = BASE_THRESH
TIERED_SENS      = SENSITIVITY


# ─────────────────────────────────────────────────────────────────────────────
# Score computation (reuses adaptive_signals; vectorised per ticker)
# ─────────────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    if isinstance(df, pd.Series):
        return df
    c = df[name]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.squeeze()


def _compute_scores_for_ticker(df: pd.DataFrame) -> pd.Series:
    """
    Compute the adaptive composite score series for one ticker.
    Imports the heavy computation from adaptive_signals.
    Returns a Series indexed like df.
    """
    # We need the full vectorised score. adaptive_signals.adaptive_latest_score
    # only returns the last two values — we need the full series for the backtester.
    # Re-implement the core scoring here using the same indicators.
    from strategy.indicators import (
        ichimoku, vidya, volume_weighted_rsi, atr, adx,
    )
    from strategy.adaptive_signals import dominant_cycle, wht_structure_score

    close  = _col(df, "Close")
    high   = _col(df, "High")
    low    = _col(df, "Low")
    volume = _col(df, "Volume")

    if len(close) < 300:
        return pd.Series(np.nan, index=df.index)

    # Adaptive Ichimoku periods via FFT
    try:
        cycle = dominant_cycle(close)
    except Exception:
        cycle = 32

    tenkan_p  = max(5,  cycle // 2)
    kijun_p   = max(10, int(cycle * 1.5))
    senkob_p  = max(20, cycle * 3)
    disp_p    = max(5,  cycle // 2)

    ich = ichimoku(high, low, close,
                   tenkan_p=tenkan_p, kijun_p=kijun_p,
                   senkob_p=senkob_p, displacement=disp_p)

    tenkan  = ich["tenkan"]
    kijun   = ich["kijun"]
    span_a  = ich["span_a"]
    span_b  = ich["span_b"]
    chikou  = ich["chikou"]

    # s1 — tenkan vs kijun crossover momentum
    tk_diff = (tenkan - kijun) / (kijun.abs() + 1e-9)
    s1 = np.tanh(tk_diff * 5)

    # s2 — lagged momentum (kijun direction)
    s2 = np.tanh((kijun - kijun.shift(kijun_p)) / (kijun.abs() + 1e-9) * 3)

    # s3 — price vs cloud
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    cloud_mid = (cloud_top + cloud_bot) / 2
    s3 = np.tanh((close - cloud_mid) / (cloud_top - cloud_bot + 1e-9) * 2)

    # s4 — chikou confirmation
    s4 = np.tanh((chikou - close.shift(disp_p)) / (close + 1e-9) * 10)

    # s5 — VIDYA trend
    vid_short = vidya(close, period=tenkan_p)
    vid_long  = vidya(close, period=kijun_p)
    s5 = np.tanh((vid_short - vid_long) / (vid_long.abs() + 1e-9) * 5)

    # s6 — volume-weighted RSI
    vrsi = volume_weighted_rsi(close, volume, period=max(7, cycle // 4))
    s6   = np.tanh((vrsi - 50) / 25)

    raw_score = s1 + s2 + s3 + s4 + s5 + s6

    # WHT multiplier
    try:
        wht_mult = wht_structure_score(close, window=128)
        wht_mult = wht_mult.clip(0.7, 1.5)
    except Exception:
        wht_mult = pd.Series(1.0, index=close.index)

    score = np.tanh(raw_score) * wht_mult
    return score.reindex(df.index)


def _atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    from strategy.indicators import atr
    high   = _col(df, "High")
    low    = _col(df, "Low")
    close  = _col(df, "Close")
    return atr(high, low, close, period=period)


# ─────────────────────────────────────────────────────────────────────────────
# Generic event-driven backtester
# ─────────────────────────────────────────────────────────────────────────────

class Position:
    __slots__ = ("ticker", "entry_price", "shares", "peak_price",
                 "stop_price", "entry_date")
    def __init__(self, ticker, entry_price, shares, stop_price, entry_date):
        self.ticker      = ticker
        self.entry_price = entry_price
        self.shares      = shares
        self.peak_price  = entry_price
        self.stop_price  = stop_price
        self.entry_date  = entry_date


def _run_backtest(
    strategy_name: str,
    ticker_data: dict[str, pd.DataFrame],
    score_data:  dict[str, pd.Series],
    entry_filter_fn,      # (ticker, date, close, score, score_prev, df) -> bool
    thresh_fn,            # (ticker, date, close, df) -> float  (the threshold to use)
) -> tuple[pd.DataFrame, dict]:
    """
    Generic event-driven backtest.

    entry_filter_fn(ticker, date, close_val, score, score_prev, df) → bool
        Return True if the stock is allowed to be entered on this bar.

    thresh_fn(ticker, date, close_val, df) → float
        Return the entry threshold for this bar.
    """
    all_dates = sorted(set(
        d for df in ticker_data.values()
        for d in df.index
        if TEST_START <= str(d)[:10] <= TEST_END
    ))

    cash       = float(INIT_CAPITAL)
    positions: dict[str, Position] = {}
    equity_curve = []
    trades_log   = []

    # Pre-slice data to test period for speed
    test_data  = {t: df.loc[df.index >= TEST_START] for t, df in ticker_data.items()}
    test_score = {t: sc.loc[sc.index >= TEST_START]  for t, sc in score_data.items()}

    for date in all_dates:
        # ── mark-to-market ───────────────────────────────────────────────────
        port_value = cash
        for t, pos in positions.items():
            df = test_data.get(t)
            if df is None or date not in df.index:
                port_value += pos.shares * pos.entry_price
                continue
            close_val = float(_col(df, "Close").loc[date])
            port_value += pos.shares * close_val

        equity_curve.append({"date": date, "equity": port_value})

        # ── trailing stop check & exit ────────────────────────────────────────
        to_close = []
        for t, pos in positions.items():
            df = test_data.get(t)
            if df is None or date not in df.index:
                continue
            close_val = float(_col(df, "Close").loc[date])
            atr_s     = _atr_series(ticker_data[t])
            atr_val   = float(atr_s.loc[date]) if date in atr_s.index else 0.0

            pos.peak_price = max(pos.peak_price, close_val)
            stop           = pos.peak_price - ATR_MULT * atr_val

            sc = test_score.get(t)
            if sc is None:
                continue
            idx = sc.index.get_loc(date) if date in sc.index else -1
            score_val = float(sc.iloc[idx]) if idx >= 0 else 0.0

            if close_val < stop or (idx >= 1 and score_val < -SCORE_THRESH):
                sell_price = close_val * (1 - SLIPPAGE)
                pnl        = (sell_price - pos.entry_price) * pos.shares
                pnl_pct    = (sell_price / pos.entry_price - 1) * 100
                cash      += pos.shares * sell_price
                trades_log.append({
                    "ticker":      t,
                    "entry_date":  pos.entry_date,
                    "exit_date":   date,
                    "entry_price": pos.entry_price,
                    "exit_price":  sell_price,
                    "shares":      pos.shares,
                    "pnl":         pnl,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": "stop" if close_val < stop else "signal",
                })
                to_close.append(t)

        for t in to_close:
            del positions[t]

        # ── entry signals ─────────────────────────────────────────────────────
        if len(positions) >= MAX_POSITIONS:
            continue

        slot_capital = port_value / MAX_POSITIONS

        for t, df in test_data.items():
            if t in positions or len(positions) >= MAX_POSITIONS:
                continue
            if date not in df.index:
                continue

            sc = test_score.get(t)
            if sc is None:
                continue
            idx = sc.index.get_loc(date) if date in sc.index else -1
            if idx < 1:
                continue

            score_val  = float(sc.iloc[idx])
            score_prev = float(sc.iloc[idx - 1])

            close_val = float(_col(df, "Close").loc[date])

            # Get the effective threshold for this bar
            eff_thresh = thresh_fn(t, date, close_val, df)
            if eff_thresh == float("inf"):
                continue

            # Fresh crossover above threshold
            if not (score_prev < eff_thresh <= score_val):
                continue

            # Custom filter (regime, etc.)
            if not entry_filter_fn(t, date, close_val, score_val, score_prev, df):
                continue

            # Position sizing: equal slot capital
            shares = max(1, int(slot_capital / close_val))
            cost   = shares * close_val * (1 + SLIPPAGE)
            if cost > cash:
                shares = max(1, int(cash / (close_val * (1 + SLIPPAGE))))
                cost   = shares * close_val * (1 + SLIPPAGE)
            if shares < 1 or cost > cash:
                continue

            atr_val    = float(_atr_series(ticker_data[t]).get(date, close_val * 0.02))
            stop_price = close_val - ATR_MULT * atr_val
            cash      -= cost
            positions[t] = Position(t, close_val, shares, stop_price, date)

    # Force-close remaining at end
    final_date = all_dates[-1]
    for t, pos in positions.items():
        df = test_data.get(t)
        close_val  = float(_col(df, "Close").iloc[-1]) if df is not None else pos.entry_price
        sell_price = close_val * (1 - SLIPPAGE)
        pnl        = (sell_price - pos.entry_price) * pos.shares
        pnl_pct    = (sell_price / pos.entry_price - 1) * 100
        cash      += pos.shares * sell_price
        trades_log.append({
            "ticker": t, "entry_date": pos.entry_date, "exit_date": final_date,
            "entry_price": pos.entry_price, "exit_price": sell_price,
            "shares": pos.shares, "pnl": pnl, "pnl_pct": pnl_pct,
            "exit_reason": "eod",
        })

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades_log)

    # ── metrics ───────────────────────────────────────────────────────────────
    eq = equity_df["equity"]
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 0.1)) - 1) * 100

    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * (252 ** 0.5)

    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max * 100
    mdd         = drawdown.min()
    calmar      = cagr / abs(mdd) if mdd != 0 else 0.0

    win_rate = 0.0
    n_trades = len(trades_df)
    if n_trades:
        win_rate = (trades_df["pnl"] > 0).mean() * 100

    metrics = {
        "strategy":    strategy_name,
        "total_return": f"{total_return:+.2f}%",
        "cagr":         f"{cagr:+.2f}%",
        "sharpe":       f"{sharpe:.2f}",
        "mdd":          f"{mdd:.2f}%",
        "calmar":       f"{calmar:.2f}",
        "win_rate":     f"{win_rate:.1f}%",
        "trades":       n_trades,
    }

    trades_df.to_csv(
        RESULTS_DIR / f"trades_{strategy_name.lower().replace(' ', '_')}.csv",
        index=False,
    )

    return equity_df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Filter functions for each strategy
# ─────────────────────────────────────────────────────────────────────────────

def _make_hard_filter(ticker_data):
    """Strategy A: hard MA100/200 gate (current sheets_trader logic)."""
    def filter_fn(ticker, date, close_val, score, score_prev, df):
        close_s = _col(df, "Close")
        ma100 = close_s.rolling(100).mean()
        ma200 = close_s.rolling(200).mean()
        if date not in ma100.index:
            return False
        m100 = float(ma100.loc[date])
        m200 = float(ma200.loc[date])
        if np.isnan(m100) or np.isnan(m200):
            return False
        return close_val > m100 * HARD_MA100_MULT and m100 > m200 * HARD_MA200_MULT
    return filter_fn


def _hard_thresh_fn(ticker, date, close_val, df):
    """Strategy A uses a fixed threshold — the hard filter is separate."""
    return SCORE_THRESH


def _make_tiered_thresh_fn(ticker_data):
    """Strategy B: threshold computed from tiered MA distance."""
    def thresh_fn(ticker, date, close_val, df):
        close_s = _col(df, "Close")
        ma100 = float(close_s.rolling(100).mean().get(date, np.nan))
        ma200 = float(close_s.rolling(200).mean().get(date, np.nan))
        return compute_effective_threshold(close_val, ma100, ma200, TIERED_BASE, TIERED_SENS)
    return thresh_fn


def _tiered_entry_fn(ticker, date, close_val, score, score_prev, df):
    """Strategy B: no separate filter — the threshold does all the work."""
    return True


def _no_filter_thresh_fn(ticker, date, close_val, df):
    return SCORE_THRESH


def _no_filter_entry_fn(ticker, date, close_val, score, score_prev, df):
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Nifty benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_nifty_benchmark() -> pd.Series:
    try:
        df = fetch_ohlcv("^NSEI", period="10y", interval="1d")
        close = _col(df, "Close")
        return close.loc[close.index >= TEST_START]
    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _plot(equity_curves: dict[str, pd.DataFrame], nifty: pd.Series, outpath: Path):
    colours = {
        "A: Hard MA filter":   "#e74c3c",
        "B: Tiered MA filter": "#00d4aa",
        "C: No MA filter":     "#f39c12",
        "Nifty 50":            "#7f8c8d",
    }

    fig = plt.figure(figsize=(14, 7), facecolor="#0d1117")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for name, eq_df in equity_curves.items():
        eq = eq_df["equity"]
        norm = eq / eq.iloc[0] * 100
        ax.plot(norm.index, norm.values, label=name,
                color=colours.get(name, "#fff"), linewidth=1.8)

    if len(nifty) > 0:
        nifty_norm = nifty / nifty.iloc[0] * 100
        ax.plot(nifty_norm.index, nifty_norm.values,
                label="Nifty 50", color=colours["Nifty 50"],
                linewidth=1.2, linestyle="--")

    ax.set_title("MA Filter Comparison — Adaptive Signal  |  2021–2025",
                 color="#eee", fontsize=13, pad=12)
    ax.set_ylabel("Equity (base = 100)", color="#aaa")
    ax.set_xlabel("Date", color="#aaa")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#eee", fontsize=9)
    ax.grid(color="#222", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved → {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    print(f"\nFetching data for {len(TICKERS)} tickers from {TRAIN_START} …")
    ticker_data: dict[str, pd.DataFrame] = {}
    failed = 0
    for i, t in enumerate(TICKERS):
        try:
            df = fetch_ohlcv(t, period="7y", interval="1d")
            if df is not None and len(df) > 300:
                ticker_data[t] = df
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(TICKERS)}  loaded={len(ticker_data)}  failed={failed}")
        time.sleep(0.05)

    print(f"\nLoaded {len(ticker_data)} tickers  ({failed} failed)")

    # ── 2. Compute scores ─────────────────────────────────────────────────────
    print("\nComputing adaptive scores … (this takes a few minutes)")
    score_data: dict[str, pd.Series] = {}
    for i, (t, df) in enumerate(ticker_data.items()):
        try:
            sc = _compute_scores_for_ticker(df)
            if sc.dropna().__len__() > 50:
                score_data[t] = sc
        except Exception as e:
            pass
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(ticker_data)}")

    print(f"  Valid scores: {len(score_data)}")

    valid = {t: ticker_data[t] for t in score_data}

    # ── 3. Run strategies ─────────────────────────────────────────────────────
    nifty = _fetch_nifty_benchmark()

    strategies = [
        ("A: Hard MA filter",   _make_hard_filter(valid),         _hard_thresh_fn),
        ("B: Tiered MA filter", _tiered_entry_fn,                  _make_tiered_thresh_fn(valid)),
        ("C: No MA filter",     _no_filter_entry_fn,               _no_filter_thresh_fn),
    ]

    equity_curves: dict[str, pd.DataFrame] = {}
    all_metrics: list[dict] = []

    for name, filter_fn, thresh_fn in strategies:
        print(f"\n  Running {name} …")
        eq, metrics = _run_backtest(name, valid, score_data, filter_fn, thresh_fn)
        equity_curves[name] = eq
        all_metrics.append(metrics)

    # ── 4. Print results table ────────────────────────────────────────────────
    col_w = {
        "strategy": 26, "total_return": 12, "cagr": 10,
        "sharpe": 8, "mdd": 10, "calmar": 8, "win_rate": 9, "trades": 7,
    }
    headers = list(col_w.keys())
    sep = "─" * (sum(col_w.values()) + len(col_w) * 3)

    print(f"\n\n{'═' * len(sep)}")
    print("  MA FILTER COMPARISON")
    print(f"  Signal: Adaptive Ichimoku + VIDYA + VRSI  |  {TEST_START[:4]}–{TEST_END[:4]}")
    print(f"  Universe: {len(valid)} tickers  |  Equal slot sizing  |  ATR stops")
    print(f"{'─' * len(sep)}")
    hdr = "  " + "  ".join(h.upper().ljust(col_w[h]) for h in headers)
    print(hdr)
    print(f"{'─' * len(sep)}")

    for m in all_metrics:
        row = "  " + "  ".join(str(m[h]).ljust(col_w[h]) for h in headers)
        print(row)

    print(f"{'═' * len(sep)}")

    # Key interpretation guide
    print("""
  INTERPRETATION
  ──────────────
  Strategy A  — Hard gate: only stocks above MA100 and MA100 > MA200*0.97
                This is what sheets_trader.py currently uses.
                Blocks ALL entries in broad corrections.

  Strategy B  — Tiered gate: threshold scales with distance below MA100
                Stocks at MA100 → threshold 0.25 (unchanged)
                5% below MA100 → threshold 0.325
                10% below MA100 → threshold 0.40
                >30% below MA100 → hard block (same as A for falling knives)
                Allows SOME entries in corrections; requires stronger signals.

  Strategy C  — No filter: pure signal, 0.25 threshold everywhere
                Upper bound on trade count; may buy into downtrends.

  Calmar = CAGR / |MDD|  — higher is better (return per unit of drawdown)
""")

    # ── 5. Chart ──────────────────────────────────────────────────────────────
    outpath = RESULTS_DIR / "ma_filter_comparison.png"
    _plot(equity_curves, nifty, outpath)

    print(f"  Trade CSVs → {RESULTS_DIR}/trades_*.csv")


if __name__ == "__main__":
    main()