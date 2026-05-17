"""
backtest/compare_strategies.py
-------------------------------
Head-to-head comparison of:
    Strategy A: Original fixed-period Ichimoku + VIDYA + VRSI
    Strategy B: Adaptive FFT + WHT Ichimoku + VIDYA + VRSI

Tested across three data regimes:
    1. Real historical NSE data (2021-2025) — the in-sample period
    2. Real bear market period (COVID crash: 2019-09 to 2020-06)
    3. Synthetic data:
        a. Declining market (drift = -0.03%/day)
        b. Pure random noise (drift = 0)

Run: python -m backtest.compare_strategies

Output:
    - Console comparison table
    - backtest/results/strategy_comparison.png
"""

import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.signals          import composite_score, MIN_BARS_NEEDED, SCORE_BUY
from strategy.adaptive_signals import adaptive_composite_score

RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Synthetic data generator ───────────────────────────────────────────────────

def make_synthetic(
    n_bars:      int   = 1500,
    n_tickers:   int   = 30,
    drift:       float = 0.0,
    vol:         float = 0.018,
    seed:        int   = 42,
    label:       str   = "synthetic",
) -> dict[str, pd.DataFrame]:
    """
    Realistic synthetic OHLCV:
        - Student-t returns (fat tails, df=4)
        - GARCH(1,1) volatility clustering
        - Market factor + idiosyncratic component
        - Overnight gaps
        - Jump shocks
    """
    dates   = pd.bdate_range("2019-01-01", periods=n_bars)
    data    = {}
    rng_mkt = np.random.default_rng(seed)

    # Market factor
    mkt_vol  = np.zeros(n_bars); mkt_vol[0] = vol
    mkt_ret  = np.zeros(n_bars)
    mkt_innov= rng_mkt.standard_t(df=4, size=n_bars) / np.sqrt(4/(4-2))
    omega, ag, bg = 0.000002, 0.09, 0.90
    for t in range(1, n_bars):
        mkt_vol[t] = np.sqrt(omega + ag*mkt_ret[t-1]**2 + bg*mkt_vol[t-1]**2)
        mkt_ret[t] = drift*0.5 + mkt_vol[t]*mkt_innov[t]

    # Market shocks
    n_sh = max(1, int(n_bars/250*5))
    for idx in rng_mkt.choice(n_bars, n_sh, replace=False):
        mkt_ret[idx] += rng_mkt.choice([-1,1]) * rng_mkt.uniform(0.02, 0.05)

    for i in range(n_tickers):
        rng_t   = np.random.default_rng(seed + i*13 + 7)
        idio_vol= np.zeros(n_bars); idio_vol[0] = vol*0.8
        idio_ret= np.zeros(n_bars)
        innov   = rng_t.standard_t(df=4, size=n_bars)/np.sqrt(4/(4-2))
        for t in range(1, n_bars):
            idio_vol[t] = np.sqrt(omega + ag*idio_ret[t-1]**2 + bg*idio_vol[t-1]**2)
            idio_ret[t] = drift*0.5 + idio_vol[t]*innov[t]

        # Stock shocks
        for idx in rng_t.choice(n_bars, rng_t.integers(2,8), replace=False):
            idio_ret[idx] += rng_t.choice([-1,1])*rng_t.uniform(0.03,0.10)

        beta     = rng_t.uniform(0.5, 1.5)
        tot_ret  = beta*mkt_ret + idio_ret
        prices   = 500*np.exp(np.cumsum(tot_ret))

        dr       = np.abs(tot_ret)*rng_t.uniform(1.5, 3.0, n_bars)
        dr       = np.clip(dr, 0.003, 0.08)
        high     = prices*(1 + dr*rng_t.uniform(0.4,1.0,n_bars))
        low      = prices*(1 - dr*rng_t.uniform(0.4,1.0,n_bars))
        gap      = rng_t.normal(0, 0.004, n_bars)
        open_    = np.roll(prices,1)*(1+gap); open_[0]=prices[0]
        open_    = np.clip(open_, low, high)
        volume   = rng_t.lognormal(np.log(1e6), 0.6, n_bars)*(1+3*np.abs(tot_ret)/vol)

        data[f"{label}_{i:03d}"] = pd.DataFrame({
            "Open": open_, "High": np.maximum(high,open_),
            "Low":  np.minimum(low,open_), "Close": prices, "Volume": volume,
        }, index=dates)

    return data


# ── Real data loader ───────────────────────────────────────────────────────────

def load_real(tickers: list[str], start: str, end: str, warmup_years: int = 2) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    warmup_start = str(pd.Timestamp(start) - pd.DateOffset(years=warmup_years))[:10]
    data = {}
    for t in tickers:
        try:
            raw = yf.download(t+".NS", start=warmup_start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:,~raw.columns.duplicated()].sort_index().dropna(subset=["Close"])
            if len(raw) >= MIN_BARS_NEEDED + 100:
                data[t] = raw
        except Exception:
            pass
        time.sleep(0.1)
    return data


# ── Signal computation ─────────────────────────────────────────────────────────

def compute_scores(data: dict, strategy: str) -> dict[str, pd.Series]:
    """Compute scores for all tickers using the specified strategy."""
    fn = composite_score if strategy == "original" else adaptive_composite_score
    scores = {}
    empty  = []
    for ticker, df in data.items():
        try:
            sc = fn(df)
            valid = sc.dropna()
            if not valid.empty:
                scores[ticker] = sc
            else:
                empty.append(f"{ticker}(len={len(df)})")
        except Exception as e:
            empty.append(f"{ticker}(err={e})")
    if empty and len(empty) <= 10:
        print(f"    No scores for: {empty}")
    elif empty:
        print(f"    {len(empty)} tickers produced no scores")
    return scores


# ── Simple event-driven backtester (no vectorbt dependency) ───────────────────

def run_backtest(
    data:       dict[str, pd.DataFrame],
    scores:     dict[str, pd.Series],
    capital:    float  = 200_000,
    threshold:  float  = SCORE_BUY,
    max_pos:    int    = 8,
    commission: float  = 0.001,
    restrict_to_dates: tuple[str,str] | None = None,
) -> dict:
    """
    Equal fractional position sizing backtester.

    Each open slot gets an equal share of current portfolio value:
        position_value = portfolio_value / max_pos

    This ensures capital is always fully deployed when signals are available,
    making percentage returns meaningful and comparable across strategies.
    ATR-based sizing is a live-trading concern (risk management), not a
    signal-quality measurement concern (which is what this backtester does).
    """
    import math

    all_dates = sorted(set(
        d for df in data.values() for d in df.index.tolist()
    ))
    all_dates = pd.DatetimeIndex(all_dates).sort_values()

    if restrict_to_dates:
        s, e  = pd.Timestamp(restrict_to_dates[0]), pd.Timestamp(restrict_to_dates[1])
        all_dates = all_dates[(all_dates >= s) & (all_dates <= e)]

    if len(all_dates) < 5:
        return {"equity": pd.Series([capital], dtype=float), "trades": []}

    cash      = capital
    positions = {}   # ticker → {shares, entry_price, peak, stop, atr, cost, score, entry_date}
    equity    = {}
    trades    = []

    for dt in all_dates:
        # ── Update trailing stops ──────────────────────────────────────────────
        for ticker, pos in list(positions.items()):
            if ticker not in data or dt not in data[ticker].index:
                continue
            price = float(data[ticker].loc[dt, "Close"])
            if price > pos["peak"]:
                pos["peak"] = price
                pos["stop"] = price - pos["atr"] * 2.5

        # ── Process exits ──────────────────────────────────────────────────────
        for ticker in list(positions.keys()):
            if ticker not in data or dt not in data[ticker].index:
                continue
            pos   = positions[ticker]
            close = float(data[ticker].loc[dt, "Close"])

            sc     = scores.get(ticker, pd.Series(dtype=float))
            s_now  = float(sc.reindex([dt], method="ffill").iloc[0]) \
                     if dt in sc.index or len(sc) else 0.0
            s_prev = float(sc.shift(1).reindex([dt], method="ffill").iloc[0]) \
                     if dt in sc.index or len(sc) else 0.0

            stop_hit   = close < pos["stop"]
            score_exit = s_now <= -threshold and s_prev > -threshold

            if stop_hit or score_exit:
                proceeds = close * pos["shares"] * (1 - commission)
                pnl      = proceeds - pos["cost"]
                cash    += proceeds
                trades.append({
                    "ticker": ticker, "entry": pos["entry_price"],
                    "exit": close, "shares": pos["shares"],
                    "pnl": pnl, "reason": "stop" if stop_hit else "score",
                    "entry_date": pos["entry_date"], "exit_date": dt,
                    "atr": pos["atr"],
                })
                del positions[ticker]

        # ── Process entries ────────────────────────────────────────────────────
        if len(positions) < max_pos:
            # Current portfolio value for sizing
            open_val = sum(
                pos["shares"] * float(data[t].loc[dt, "Close"])
                for t, pos in positions.items()
                if t in data and dt in data[t].index
            )
            port_val = cash + open_val

            candidates = []
            for ticker in scores:
                if ticker in positions or ticker not in data:
                    continue
                if dt not in data[ticker].index:
                    continue
                sc = scores[ticker]
                if dt not in sc.index:
                    continue
                s_now  = float(sc.loc[dt])
                s_prev = float(sc.shift(1).loc[dt]) if dt in sc.index else 0.0
                if s_now >= threshold and s_prev < threshold:
                    candidates.append((ticker, s_now))

            candidates.sort(key=lambda x: -x[1])
            slots = max_pos - len(positions)

            for ticker, score_val in candidates[:slots]:
                close = float(data[ticker].loc[dt, "Close"])

                # ── Momentum quality filter ────────────────────────────────────
                # Require close > MA100 and MA100 > MA200 * 0.97
                # Rejects bounces in structural downtrends
                try:
                    hist  = data[ticker][data[ticker].index <= dt]
                    c_ser = hist["Close"] if isinstance(hist["Close"], pd.Series) \
                            else hist["Close"].iloc[:, 0]
                    if len(c_ser) >= 200:
                        ma100 = float(c_ser.rolling(100).mean().iloc[-1])
                        ma200 = float(c_ser.rolling(200).mean().iloc[-1])
                        if not (close > ma100 and ma100 > ma200 * 0.97):
                            continue   # skip — structural downtrend
                except Exception:
                    pass  # insufficient history — allow through

                # ── Position sizing ────────────────────────────────────────
                # Capital allocation: equal slot per position
                # ATR determines stop placement, not share count
                # This mirrors real trading: allocate capital evenly,
                # use ATR to know when you're wrong (stop), not how much to buy
                slot_capital = port_val / max_pos
                shares       = max(1, math.floor(slot_capital / close))
                cost         = close * shares * (1 + commission)

                if cost > cash:
                    shares = max(0, math.floor(
                        cash * 0.98 / close / (1 + commission)
                    ))
                    if shares < 1:
                        continue
                    cost = close * shares * (1 + commission)

                # ATR for trailing stop
                past  = data[ticker][data[ticker].index <= dt].tail(20)
                h = past["High"] if isinstance(past["High"], pd.Series) else past["High"].iloc[:,0]
                l = past["Low"]  if isinstance(past["Low"],  pd.Series) else past["Low"].iloc[:,0]
                c = past["Close"]if isinstance(past["Close"],pd.Series)else past["Close"].iloc[:,0]
                tr    = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
                atr_v = float(tr.ewm(span=14,adjust=False).mean().iloc[-1])

                cash -= cost
                positions[ticker] = {
                    "shares":      shares,
                    "entry_price": close,
                    "entry_date":  dt,
                    "peak":        close,
                    "stop":        close - atr_v * 2.5,
                    "atr":         atr_v,
                    "score":       score_val,
                    "cost":        cost,
                }

        # ── Mark to market ─────────────────────────────────────────────────────
        open_val = sum(
            pos["shares"] * float(data[t].loc[dt, "Close"])
            for t, pos in positions.items()
            if t in data and dt in data[t].index
        )
        equity[dt] = cash + open_val

    return {"equity": pd.Series(equity).sort_index(), "trades": trades}


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(result: dict, capital: float, label: str) -> dict:
    eq     = result["equity"].dropna()
    trades = result["trades"]

    if len(eq) < 2:
        return {"label": label, "return": 0, "cagr": 0, "sharpe": 0,
                "mdd": 0, "win_rate": 0, "n_trades": 0, "avg_risk_pct": 0}

    total_ret = float(eq.iloc[-1]/eq.iloc[0] - 1)
    years     = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr      = float((eq.iloc[-1]/eq.iloc[0])**(1/years) - 1) if years > 0 else 0
    daily_ret = eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean()/daily_ret.std()*np.sqrt(252)) if daily_ret.std()>0 else 0
    roll_max  = eq.cummax()
    mdd       = float(((eq - roll_max)/roll_max).min())
    pnls      = [t["pnl"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    win_rate  = len(wins)/len(pnls) if pnls else 0

    # Average capital at risk per trade (stop distance as % of position value)
    risk_pcts = []
    for t in trades:
        if t["entry"] > 0 and "atr" in t:
            risk_pcts.append(t["atr"] * 2.5 / t["entry"])
    avg_risk_pct = float(np.mean(risk_pcts)) if risk_pcts else 0

    return {
        "label":        label,
        "return":       total_ret,
        "cagr":         cagr,
        "sharpe":       sharpe,
        "mdd":          mdd,
        "win_rate":     win_rate,
        "n_trades":     len(trades),
        "avg_risk_pct": avg_risk_pct,
        "equity":       eq,
    }


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_comparison(all_results: list[dict], save_path: Path):
    """
    4-panel chart: one panel per test regime, both strategies overlaid.
    """
    regimes = ["Real Historical", "Bear Market", "Declining Synthetic", "Random Noise"]
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0a0a14")
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28)

    colors = {"original": "#00d4aa", "adaptive": "#ffaa00"}

    for idx, regime in enumerate(regimes):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#222233")
        ax.set_title(regime, color="#ffffff", fontsize=10, pad=8, fontweight="bold")
        ax.set_ylabel("Portfolio (indexed)", color="#888888", fontsize=8)

        for strat in ["original", "adaptive"]:
            key = f"{regime}_{strat}"
            r   = next((x for x in all_results if x.get("key") == key), None)
            if r is None or "equity" not in r:
                continue

            eq    = r["equity"]
            eq_n  = eq / eq.iloc[0] * 100
            ret   = r["return"]
            sh    = r["sharpe"]
            label = f"{'Original' if strat=='original' else 'Adaptive FFT+WHT'} ({ret:+.1%}, Sharpe {sh:.2f})"
            ls    = "-" if strat == "original" else "--"
            ax.plot(eq_n.index, eq_n.values, color=colors[strat],
                    linewidth=1.6, linestyle=ls, label=label, alpha=0.9)

        ax.axhline(100, color="#333344", linewidth=0.8, linestyle=":")
        ax.legend(facecolor="#111118", labelcolor="#aaaaaa", fontsize=7,
                  framealpha=0.8, loc="upper left")

    fig.suptitle(
        "Strategy Comparison: Original vs Adaptive FFT+WHT Ichimoku",
        color="#ffffff", fontsize=14, fontweight="bold", y=0.98,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved → {save_path}")


# ── Print table ────────────────────────────────────────────────────────────────

def print_table(all_results: list[dict]):
    regimes = ["Real Historical", "Bear Market", "Declining Synthetic", "Random Noise"]
    print(f"\n{'═'*88}")
    print(f"  STRATEGY COMPARISON  |  Equal slot capital + ATR stops + MA100/200 filter")
    print(f"  Entry requires: score crossover AND close > MA100 > MA200×0.97")
    print(f"{'─'*88}")
    print(f"  {'Regime':<24}  {'Strategy':<14}  {'Return':>8}  {'CAGR':>7}  "
          f"{'Sharpe':>7}  {'MDD':>7}  {'WinRate':>8}  {'Trades':>7}  {'AvgRisk':>8}")
    print(f"{'─'*88}")

    for regime in regimes:
        for strat in ["original", "adaptive"]:
            key = f"{regime}_{strat}"
            r   = next((x for x in all_results if x.get("key") == key), None)
            if r is None:
                continue
            name = "Original" if strat == "original" else "Adaptive"
            print(f"  {regime:<24}  {name:<14}  "
                  f"{r['return']:>+8.2%}  {r['cagr']:>+7.2%}  "
                  f"{r['sharpe']:>7.2f}  {r['mdd']:>7.2%}  "
                  f"{r['win_rate']:>8.1%}  {r['n_trades']:>7d}  "
                  f"{r.get('avg_risk_pct',0):>8.1%}")
        print(f"{'─'*88}")

    print(f"{'═'*88}")
    print("""
  Sizing: each position = portfolio_value / max_positions shares
  Stop:   entry_price - 2.5 × ATR14  (ATR determines exit, not entry size)
  AvgRisk = avg stop distance as % of position value per trade
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NIFTY_SAMPLE = [
        # IT
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE",
        # Financials
        "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN",
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
        # Consumer
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
        "DABUR", "MARICO", "COLPAL",
        # Pharma
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
        # Auto
        "MARUTI", "TATAMOTORS", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "TVSMOTOR",
        # Industrials / Infra
        "LT", "POWERGRID", "NTPC", "ONGC", "COALINDIA", "BPCL",
        "ADANIPORTS", "ADANIENT",
        # Materials
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "ULTRACEMCO",
        # Consumer Discretionary
        "TITAN", "ASIANPAINT", "DMART",
    ]

    CAPITAL = 200_000
    ORIG_THRESH  = 0.40   # original fixed threshold
    # Adaptive uses the score value that corresponds to the same
    # percentile of signals as the original's 0.40 threshold.
    # We calibrate this empirically after computing scores.
    ADAPT_THRESH = 0.25   # starting value — will be auto-calibrated below
    all_results = []

    # ── 1. Real historical data (2021-2025) ────────────────────────────────────
    print("\n" + "═"*60)
    print("  Loading real historical data (2021-2025) ...")
    # Load 4yr warmup so adaptive strategy has enough bars for FFT+WHT+Ichimoku
    real_hist = load_real(NIFTY_SAMPLE, "2021-01-01", "2025-05-01", warmup_years=4)
    print(f"  Loaded {len(real_hist)} tickers")
    if real_hist:
        sample = next(iter(real_hist.values()))
        print(f"  Sample bar count: {len(sample)} (need ~{128+64+120+50}=362 for adaptive)")

    for strat in ["original", "adaptive"]:
        print(f"  Computing {strat} scores ...")
        scores = compute_scores(real_hist, strat)
        if strat == "adaptive" and scores:
            import numpy as np
            all_vals = np.concatenate([s.dropna().values for s in scores.values()])
            # Calibrate: find threshold that gives same % of bars above as original
            # Original: what % of bars are above 0.40?
            orig_vals = []
            orig_scores_tmp = compute_scores(real_hist, "original")
            if orig_scores_tmp:
                orig_vals = np.concatenate([s.dropna().values for s in orig_scores_tmp.values()])
                orig_pct = (orig_vals >= ORIG_THRESH).mean()
                # Set adaptive threshold to same percentile
                ADAPT_THRESH = float(np.percentile(all_vals, (1 - orig_pct) * 100))
                ADAPT_THRESH = max(0.15, min(0.45, ADAPT_THRESH))  # sanity clamp
            print(f"    Score stats: min={all_vals.min():.3f} max={all_vals.max():.3f} "
                  f"mean={all_vals.mean():.3f} std={all_vals.std():.3f}")
            above = (all_vals >= ADAPT_THRESH).mean() * 100
            print(f"    Auto-calibrated threshold: {ADAPT_THRESH:.3f} "
                  f"({above:.2f}% bars above)")
            crossovers = sum(
                int(((s.dropna() >= ADAPT_THRESH) & (s.dropna().shift(1) < ADAPT_THRESH)).sum())
                for s in scores.values()
            )
            print(f"    Entry crossovers: {crossovers}")
        thresh = ORIG_THRESH if strat == "original" else ADAPT_THRESH
        print(f"  Running backtest (threshold={thresh}) ...")
        result = run_backtest(
            real_hist, scores, capital=CAPITAL,
            threshold=thresh,
            restrict_to_dates=("2021-01-01", "2025-05-01"),
        )
        m = metrics(result, CAPITAL, strat)
        m["key"] = f"Real Historical_{strat}"
        all_results.append(m)
        print(f"  {strat}: return={m['return']:+.2%} sharpe={m['sharpe']:.2f}")

    # ── 2. Bear market (COVID crash: Sep 2019 – Jun 2020) ─────────────────────
    print("\n" + "═"*60)
    print("  Loading bear market data (2019-09 to 2020-06) ...")
    bear_data = load_real(NIFTY_SAMPLE, "2019-09-01", "2020-06-30", warmup_years=4)
    print(f"  Loaded {len(bear_data)} tickers")

    for strat in ["original", "adaptive"]:
        print(f"  Computing {strat} scores ...")
        scores = compute_scores(bear_data, strat)
        thresh = ORIG_THRESH if strat == "original" else ADAPT_THRESH
        result = run_backtest(
            bear_data, scores, capital=CAPITAL,
            threshold=thresh,
            restrict_to_dates=("2019-09-01", "2020-06-30"),
        )
        m = metrics(result, CAPITAL, strat)
        m["key"] = f"Bear Market_{strat}"
        all_results.append(m)
        print(f"  {strat}: return={m['return']:+.2%} sharpe={m['sharpe']:.2f}")

    # ── 3. Synthetic declining market ─────────────────────────────────────────
    print("\n" + "═"*60)
    print("  Generating synthetic declining data ...")
    declining = make_synthetic(n_bars=1500, n_tickers=20,
                               drift=-0.0003, label="declining", seed=42)

    for strat in ["original", "adaptive"]:
        print(f"  Computing {strat} scores ...")
        scores = compute_scores(declining, strat)
        thresh = ORIG_THRESH if strat == "original" else ADAPT_THRESH
        result = run_backtest(declining, scores, capital=CAPITAL, threshold=thresh)
        m = metrics(result, CAPITAL, strat)
        m["key"] = f"Declining Synthetic_{strat}"
        all_results.append(m)
        print(f"  {strat}: return={m['return']:+.2%} sharpe={m['sharpe']:.2f}")

    # ── 4. Synthetic random noise ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  Generating synthetic random noise data ...")
    random = make_synthetic(n_bars=1500, n_tickers=20,
                            drift=0.0, label="random", seed=999)

    for strat in ["original", "adaptive"]:
        print(f"  Computing {strat} scores ...")
        scores = compute_scores(random, strat)
        thresh = ORIG_THRESH if strat == "original" else ADAPT_THRESH
        result = run_backtest(random, scores, capital=CAPITAL, threshold=thresh)
        m = metrics(result, CAPITAL, strat)
        m["key"] = f"Random Noise_{strat}"
        all_results.append(m)
        print(f"  {strat}: return={m['return']:+.2%} sharpe={m['sharpe']:.2f}")

    # ── Results ────────────────────────────────────────────────────────────────
    print_table(all_results)
    plot_comparison(all_results, RESULTS_DIR / "strategy_comparison.png")