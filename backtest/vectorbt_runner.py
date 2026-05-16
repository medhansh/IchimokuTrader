"""
backtest/vectorbt_runner.py
----------------------------
Vectorbt backtesting. This is the ONLY file that imports vectorbt.

Install dev deps: pip install -r requirements-dev.txt
Run: python -m backtest.vectorbt_runner

Produces:
    - Strategy comparison table (threshold sensitivity)
    - Equity curve chart saved to backtest/results/
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.tickers     import NIFTY500
from strategy.signals import composite_score, signals_for_backtest, size_position
from strategy.indicators import atr

RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
START      = "2021-01-01"
END        = "2025-05-01"
CAPITAL    = 25_000_000   # ₹2.5Cr — sized to allow 1 share of most expensive stock
COMMISSION = 0.001
SLIPPAGE   = 0.001


# ── Data loading ───────────────────────────────────────────────────────────────

def load_close(tickers, start, end):
    """Wide DataFrame: dates × tickers (Close prices)."""
    frames = {}
    for i, t in enumerate(tickers):
        try:
            raw = yf.download(t + ".NS", start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]
            if len(raw) > 300:
                frames[t] = raw["Close"].rename(t)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} loaded")
        time.sleep(0.08)
    df = pd.DataFrame(frames).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def load_all_ohlcv(tickers, start, end):
    """Dict of ticker → full OHLCV DataFrame."""
    data = {}
    for i, t in enumerate(tickers):
        try:
            raw = yf.download(t + ".NS", start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()].sort_index()
            if len(raw) > 300:
                data[t] = raw
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} fetched")
        time.sleep(0.08)
    return data


# ── Score computation ──────────────────────────────────────────────────────────

def build_score_df(data: dict) -> pd.DataFrame:
    scores = {}
    for t, df in data.items():
        try:
            sc = composite_score(df).dropna()
            scores[t] = sc
        except Exception:
            pass
    df = pd.DataFrame(scores).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


# ── Dynamic sizing ─────────────────────────────────────────────────────────────

def build_size_df(
    score_df:  pd.DataFrame,
    close_df:  pd.DataFrame,
    data:      dict,
    threshold: float,
    capital:   float,
) -> pd.DataFrame:
    """Size DataFrame: share count at each entry bar, 0 elsewhere."""
    entries, _ = signals_for_backtest(score_df, buy_thresh=threshold, sell_thresh=-threshold)

    atr_frames = {}
    for t in score_df.columns:
        if t in data:
            atr_frames[t] = atr(data[t]).rename(t)
    atr_df = pd.DataFrame(atr_frames).reindex_like(close_df).ffill()

    size_df = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    for t in score_df.columns:
        if t not in close_df.columns or t not in atr_df.columns:
            continue
        for dt in entries.index[entries[t]]:
            if dt not in close_df.index:
                continue
            cl  = float(close_df.loc[dt, t])
            av  = float(atr_df.loc[dt, t]) if not pd.isna(atr_df.loc[dt, t]) else 0
            sc  = float(score_df.loc[dt, t]) if not pd.isna(score_df.loc[dt, t]) else threshold
            sh  = size_position(cl, av, sc, capital, threshold=threshold)
            size_df.loc[dt, t] = float(sh)

    return size_df


# ── Portfolio builder ──────────────────────────────────────────────────────────

def build_portfolio(close_df, score_df, size_df, threshold, capital):
    entries, exits = signals_for_backtest(
        score_df.reindex_like(close_df),
        buy_thresh=threshold, sell_thresh=-threshold,
    )
    # Detect valid size_type
    try:
        from vectorbt.portfolio.enums import SizeType
        fields = [f.lower() for f in SizeType._fields_]
        st = "shares" if "shares" in fields else ("amount" if "amount" in fields else fields[0])
    except Exception:
        st = None

    kw = dict(
        close=close_df, entries=entries, exits=exits, size=size_df,
        fees=COMMISSION, slippage=SLIPPAGE,
        init_cash=capital * 2,
        freq="1D", group_by=True, cash_sharing=True,
    )
    if st:
        kw["size_type"] = st

    return vbt.Portfolio.from_signals(**kw)


# ── Benchmark ──────────────────────────────────────────────────────────────────

def nifty_return(start, end):
    try:
        raw = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        c = raw["Close"].dropna()
        return float(c.iloc[-1] / c.iloc[0] - 1)
    except Exception:
        return 0.0


# ── Sensitivity sweep ──────────────────────────────────────────────────────────

def threshold_sweep(close_df, score_df, size_dfs, thresholds, capital, benchmark):
    print(f"\n{'─'*70}")
    print(f"  {'thresh':>7}  {'return':>9}  {'vs_nifty':>9}  "
          f"{'sharpe':>7}  {'mdd':>8}  {'trades':>7}  {'win_rate':>9}")
    print(f"{'─'*70}")

    rows = []
    for thresh in thresholds:
        try:
            pf      = build_portfolio(close_df, score_df, size_dfs[thresh], thresh, capital)
            ret     = float(pf.total_return())
            sharpe  = float(pf.sharpe_ratio())
            mdd     = float(pf.max_drawdown())
            trades  = pf.trades.records_readable
            n       = len(trades)
            wr      = float((trades["PnL"] > 0).mean()) if n > 0 else 0

            print(f"  {thresh:>7.2f}  {ret:>+9.2%}  {ret-benchmark:>+9.2%}  "
                  f"{sharpe:>7.2f}  {mdd:>8.2%}  {n:>7d}  {wr:>9.1%}")
            rows.append(dict(threshold=thresh, total_return=ret, vs_nifty=ret-benchmark,
                             sharpe=sharpe, max_drawdown=mdd, n_trades=n, win_rate=wr))
        except Exception as e:
            print(f"  {thresh:>7.2f}  failed: {e}")

    print(f"{'─'*70}")
    return pd.DataFrame(rows)


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_equity(pf, benchmark_ret, threshold, start, end):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")

    equity = pf.value()
    eq_n   = equity / equity.iloc[0] * 100
    ret    = float(pf.total_return())
    sharpe = float(pf.sharpe_ratio())

    axes[0].plot(eq_n.index, eq_n.values, color="#00d4aa", linewidth=1.8,
                 label=f"Strategy ({ret:+.1%})")
    roll_max = eq_n.cummax()
    axes[0].fill_between(eq_n.index, eq_n.values, roll_max.values,
                         alpha=0.15, color="#ff4444", label="Drawdown")
    axes[0].axhline(100, color="#444", linewidth=0.7, linestyle="--")
    axes[0].set_title(
        f"Ichimoku + VIDYA + VRSI  |  Nifty 500  |  {start}→{end}  "
        f"|  thresh={threshold}  |  Return {ret:+.1%}  |  Sharpe {sharpe:.2f}",
        color="#ffffff", pad=10,
    )
    axes[0].set_ylabel("Portfolio (indexed)", color="#aaaaaa")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="#aaaaaa")

    trades = pf.trades.records_readable
    if not trades.empty and "PnL" in trades.columns:
        colors = ["#00ff88" if p > 0 else "#ff4444" for p in trades["PnL"]]
        axes[1].bar(range(len(trades)), trades["PnL"].values,
                    color=colors, alpha=0.7)
        axes[1].axhline(0, color="#555", linewidth=0.7)
        axes[1].set_ylabel("Trade PnL (₹)", color="#aaaaaa")

    plt.tight_layout(pad=1.5)
    path = RESULTS_DIR / f"equity_thresh{threshold}_{start[:4]}_{end[:4]}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Chart saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.66]
    BEST_THRESH = 0.40

    print(f"\nIchimoku + VIDYA + VRSI Backtest")
    print(f"Period: {START} → {END}  |  Tickers: {len(NIFTY500)}")
    print(f"\nLoading OHLCV data ...")
    data = load_all_ohlcv(NIFTY500, START, END)
    print(f"Loaded {len(data)} tickers")

    print("\nBuilding close DataFrame ...")
    close_df = pd.DataFrame(
        {t: df["Close"].rename(t) for t, df in data.items()}
    ).sort_index()

    print("Computing composite scores ...")
    score_df = build_score_df(data)

    # Align
    common_idx  = close_df.index.intersection(score_df.index)
    common_cols = [c for c in close_df.columns if c in score_df.columns]
    close_df  = close_df.loc[common_idx, common_cols]
    score_df  = score_df.loc[common_idx, common_cols]

    print("Building size DataFrames for each threshold ...")
    size_dfs = {}
    for thresh in THRESHOLDS:
        print(f"  thresh={thresh} ...")
        size_dfs[thresh] = build_size_df(score_df, close_df, data, thresh, CAPITAL)

    benchmark = nifty_return(START, END)
    print(f"\nNifty 50 benchmark: {benchmark:+.2%}")

    # Threshold sweep
    results = threshold_sweep(close_df, score_df, size_dfs, THRESHOLDS, CAPITAL, benchmark)
    results.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)

    # Chart for best threshold
    print(f"\nGenerating chart for thresh={BEST_THRESH} ...")
    pf = build_portfolio(close_df, score_df, size_dfs[BEST_THRESH], BEST_THRESH, CAPITAL)
    plot_equity(pf, benchmark, BEST_THRESH, START, END)