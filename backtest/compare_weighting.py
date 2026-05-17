"""
backtest/compare_weighting.py
-------------------------------
Two-strategy comparison:

    A: Adaptive baseline
       Fixed threshold 0.25 throughout.

    B: Adaptive + portfolio-level confidence
       Same score as A. Entry threshold rises when the portfolio has had
       a bad recent week, falls back to 0.25 when performance is neutral
       or positive. The score itself is unchanged — only the bar for entry
       is adaptive.

       threshold(t) = 0.25 + max(0, -recent_5day_return) * 2.0
       clamped to [0.25, 0.45]

Both strategies:
    - Equal slot capital sizing (portfolio / max_positions)
    - ATR trailing stop (2.5 × ATR14)
    - MA100/200 trend filter (close > MA100 > MA200 × 0.97)

Four test regimes:
    1. Real historical NSE (2021-2025)
    2. Real bear market (COVID crash 2019-2020)
    3. Synthetic declining (-7% annual drift, GARCH + fat tails)
    4. Synthetic random noise (zero drift)

Run: python -m backtest.compare_weighting
"""

import sys
import time
import logging
import warnings
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.adaptive_signals       import adaptive_composite_score
from strategy.portfolio_confidence   import PortfolioThreshold

RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

COMMISSION  = 0.001
MAX_POS     = 8
BASE_THRESH = 0.25

NIFTY_SAMPLE = [
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "COFORGE",
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN",
    "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "MARUTI", "TATAMOTORS", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR",
    "LT", "POWERGRID", "NTPC", "ONGC", "COALINDIA",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "ULTRACEMCO",
    "TITAN", "ASIANPAINT", "DMART", "RELIANCE", "ADANIPORTS",
]


# ── Data ───────────────────────────────────────────────────────────────────────

def load_real(tickers, start, end, warmup_years=4):
    import yfinance as yf
    warmup = str(pd.Timestamp(start) - pd.DateOffset(years=warmup_years))[:10]
    data   = {}
    for t in tickers:
        try:
            raw = yf.download(t+".NS", start=warmup, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:,~raw.columns.duplicated()].sort_index().dropna(subset=["Close"])
            if len(raw) >= 400:
                data[t] = raw
        except Exception:
            pass
        time.sleep(0.08)
    log.info(f"Loaded {len(data)}/{len(tickers)} tickers")
    return data


def make_synthetic(n_bars=1500, n_tickers=20, drift=0.0,
                   vol=0.018, seed=42, label="synth"):
    dates   = pd.bdate_range("2019-01-01", periods=n_bars)
    data    = {}
    rng_mkt = np.random.default_rng(seed)
    mv = np.zeros(n_bars); mv[0] = vol
    mr = np.zeros(n_bars)
    mi = rng_mkt.standard_t(df=4, size=n_bars) / np.sqrt(2)
    omega, ag, bg = 0.000002, 0.09, 0.90
    for t in range(1, n_bars):
        mv[t] = np.sqrt(omega + ag*mr[t-1]**2 + bg*mv[t-1]**2)
        mr[t] = drift*0.5 + mv[t]*mi[t]
    for idx in rng_mkt.choice(n_bars, max(1, n_bars//250*5), replace=False):
        mr[idx] += rng_mkt.choice([-1,1]) * rng_mkt.uniform(0.02, 0.05)
    for i in range(n_tickers):
        rng_t = np.random.default_rng(seed+i*13+7)
        iv = np.zeros(n_bars); iv[0]=vol*0.8
        ir = np.zeros(n_bars)
        inn= rng_t.standard_t(df=4, size=n_bars)/np.sqrt(2)
        for t in range(1, n_bars):
            iv[t] = np.sqrt(omega+ag*ir[t-1]**2+bg*iv[t-1]**2)
            ir[t] = drift*0.5+iv[t]*inn[t]
        for idx in rng_t.choice(n_bars, rng_t.integers(2,8), replace=False):
            ir[idx] += rng_t.choice([-1,1])*rng_t.uniform(0.03,0.10)
        beta  = rng_t.uniform(0.5, 1.5)
        tot   = beta*mr + ir
        px    = 500*np.exp(np.cumsum(tot))
        dr    = np.clip(np.abs(tot)*rng_t.uniform(1.5,3.0,n_bars), 0.003, 0.08)
        hi    = px*(1+dr*rng_t.uniform(0.4,1.0,n_bars))
        lo    = px*(1-dr*rng_t.uniform(0.4,1.0,n_bars))
        gap   = rng_t.normal(0, 0.004, n_bars)
        op    = np.roll(px,1)*(1+gap); op[0]=px[0]
        vols  = rng_t.lognormal(np.log(1e6),0.6,n_bars)*(1+3*np.abs(tot)/vol)
        data[f"{label}_{i:03d}"] = pd.DataFrame({
            "Open":op,"High":np.maximum(hi,op),
            "Low":np.minimum(lo,op),"Close":px,"Volume":vols,
        }, index=dates)
    return data


# ── Score computation ──────────────────────────────────────────────────────────

def compute_scores(data: dict) -> dict:
    """Compute adaptive scores for all tickers. Same scores for both strategies."""
    scores = {}
    for ticker, df in data.items():
        try:
            sc = adaptive_composite_score(df)
            if not sc.dropna().empty:
                scores[ticker] = sc
        except Exception as e:
            log.debug(f"  {ticker}: {e}")
    log.info(f"  Scored {len(scores)} tickers")
    return scores


# ── MA filter ─────────────────────────────────────────────────────────────────

def passes_ma_filter(data: dict, ticker: str, dt) -> bool:
    """MA100/200 trend filter — same as sheets_trader."""
    try:
        hist  = data[ticker][data[ticker].index <= dt]
        c_ser = hist["Close"] if isinstance(hist["Close"], pd.Series) \
                else hist["Close"].iloc[:,0]
        if len(c_ser) < 200:
            return True   # insufficient history — allow through
        ma100    = float(c_ser.rolling(100).mean().iloc[-1])
        ma200    = float(c_ser.rolling(200).mean().iloc[-1])
        close_now= float(data[ticker].loc[dt, "Close"])
        return close_now > ma100 and ma100 > ma200 * 0.97
    except Exception:
        return True


# ── Backtester ─────────────────────────────────────────────────────────────────

def run_backtest(data: dict, scores: dict,
                 capital:   float = 200_000,
                 use_adaptive_threshold: bool = False,
                 max_pos:   int   = MAX_POS,
                 commission:float = COMMISSION,
                 restrict_to_dates=None) -> dict:
    """
    Event-driven backtester.

    use_adaptive_threshold=False → fixed threshold 0.25 (baseline)
    use_adaptive_threshold=True  → PortfolioThreshold adjusts based on
                                   recent portfolio return (adaptive)
    """
    all_dates = pd.DatetimeIndex(sorted({
        d for df in data.values() for d in df.index.tolist()
    })).sort_values()

    if restrict_to_dates:
        s, e  = pd.Timestamp(restrict_to_dates[0]), pd.Timestamp(restrict_to_dates[1])
        all_dates = all_dates[(all_dates >= s) & (all_dates <= e)]

    if len(all_dates) < 5:
        return {"equity": pd.Series([capital], dtype=float), "trades": [],
                "thresholds": []}

    cash       = capital
    positions  = {}
    equity     = {}
    trades     = []
    thresholds = []    # track threshold over time for inspection

    pt = PortfolioThreshold() if use_adaptive_threshold else None

    for dt in all_dates:
        # ── Current portfolio value ────────────────────────────────────────────
        open_val = sum(
            pos["shares"] * float(data[t].loc[dt, "Close"])
            for t, pos in positions.items()
            if t in data and dt in data[t].index
        )
        port_val = cash + open_val

        # Update portfolio threshold tracker
        if pt is not None:
            pt.update(port_val)
            threshold = pt.current_threshold()
        else:
            threshold = BASE_THRESH
        thresholds.append(threshold)

        # ── Update trailing stops ──────────────────────────────────────────────
        for t, pos in list(positions.items()):
            if t in data and dt in data[t].index:
                price = float(data[t].loc[dt, "Close"])
                if price > pos["peak"]:
                    pos["peak"] = price
                    pos["stop"] = price - pos["atr"] * 2.5

        # ── Exits ──────────────────────────────────────────────────────────────
        for t in list(positions.keys()):
            if t not in data or dt not in data[t].index:
                continue
            pos   = positions[t]
            close = float(data[t].loc[dt, "Close"])
            sc    = scores.get(t, pd.Series(dtype=float))
            s_now = float(sc.reindex([dt], method="ffill").iloc[0]) \
                    if len(sc) and dt in sc.index else 0.0
            s_prv = float(sc.shift(1).reindex([dt], method="ffill").iloc[0]) \
                    if len(sc) and dt in sc.index else 0.0

            stop_hit   = close < pos["stop"]
            # Exit uses baseline threshold (don't raise exit bar — let stops work)
            score_exit = s_now <= -BASE_THRESH and s_prv > -BASE_THRESH

            if stop_hit or score_exit:
                proceeds = close * pos["shares"] * (1 - commission)
                trades.append({
                    "pnl":        proceeds - pos["cost"],
                    "entry":      pos["entry_price"],
                    "exit":       close,
                    "atr":        pos["atr"],
                    "entry_date": pos["entry_date"],
                    "exit_date":  dt,
                })
                cash += proceeds
                del positions[t]

        # ── Entries ────────────────────────────────────────────────────────────
        if len(positions) < max_pos:
            cands = []
            for t in scores:
                if t in positions or t not in data or dt not in data[t].index:
                    continue
                sc = scores[t]
                if dt not in sc.index:
                    continue
                s_now = float(sc.loc[dt])
                s_prv = float(sc.shift(1).loc[dt]) if dt in sc.index else 0.0

                # Entry crossover uses adaptive threshold
                if not (s_now >= threshold and s_prv < threshold):
                    continue
                if not passes_ma_filter(data, t, dt):
                    continue
                cands.append((t, s_now))

            cands.sort(key=lambda x: -x[1])
            for t, _ in cands[:max_pos - len(positions)]:
                close  = float(data[t].loc[dt, "Close"])
                shares = max(1, math.floor(port_val / max_pos / close))
                cost   = close * shares * (1 + commission)
                if cost > cash:
                    shares = max(0, math.floor(cash*0.98/close/(1+commission)))
                    if shares < 1:
                        continue
                    cost = close * shares * (1 + commission)

                past  = data[t][data[t].index <= dt].tail(20)
                h = past["High"] if isinstance(past["High"],pd.Series) else past["High"].iloc[:,0]
                l = past["Low"]  if isinstance(past["Low"], pd.Series) else past["Low"].iloc[:,0]
                c = past["Close"]if isinstance(past["Close"],pd.Series)else past["Close"].iloc[:,0]
                tr    = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
                atr_v = float(tr.ewm(span=14,adjust=False).mean().iloc[-1])

                cash -= cost
                positions[t] = {
                    "shares":      shares,
                    "entry_price": close,
                    "entry_date":  dt,
                    "peak":        close,
                    "stop":        close - atr_v * 2.5,
                    "atr":         atr_v,
                    "cost":        cost,
                }

        # ── Mark to market ─────────────────────────────────────────────────────
        equity[dt] = cash + sum(
            pos["shares"] * float(data[t].loc[dt, "Close"])
            for t, pos in positions.items()
            if t in data and dt in data[t].index
        )

    return {
        "equity":     pd.Series(equity).sort_index(),
        "trades":     trades,
        "thresholds": pd.Series(thresholds, index=all_dates[:len(thresholds)]),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(result: dict, label: str) -> dict:
    eq     = result["equity"].dropna()
    trades = result["trades"]
    if len(eq) < 2:
        return {"label":label,"return":0,"cagr":0,"sharpe":0,
                "mdd":0,"calmar":0,"win_rate":0,"n_trades":0}
    ret  = float(eq.iloc[-1]/eq.iloc[0] - 1)
    yrs  = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = float((eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1) if yrs > 0 else 0
    dr   = eq.pct_change().dropna()
    sh   = float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std() > 0 else 0
    mdd  = float(((eq-eq.cummax())/eq.cummax()).min())
    pnls = [t["pnl"] for t in trades]
    wr   = len([p for p in pnls if p>0])/len(pnls) if pnls else 0
    cal  = abs(cagr/mdd) if mdd != 0 else 0
    return {"label":label,"return":ret,"cagr":cagr,"sharpe":sh,
            "mdd":mdd,"calmar":cal,"win_rate":wr,"n_trades":len(trades),
            "equity":eq,"thresholds":result.get("thresholds",pd.Series())}


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_results(results_by_regime: dict, save_path: Path):
    regimes = list(results_by_regime.keys())
    # 2 rows: equity curves top, threshold evolution bottom (real data only)
    fig = plt.figure(figsize=(7*len(regimes), 10))
    fig.patch.set_facecolor("#0a0a14")
    gs  = gridspec.GridSpec(2, len(regimes), hspace=0.35, wspace=0.22,
                            height_ratios=[3, 1])

    colors = {
        "Adaptive (baseline)":               "#00d4aa",
        "Adaptive + portfolio confidence":   "#ffaa00",
    }
    styles = {
        "Adaptive (baseline)":               "-",
        "Adaptive + portfolio confidence":   "--",
    }

    for idx, regime in enumerate(regimes):
        # ── Equity curve ───────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#222233")
        ax.set_title(regime, color="#ffffff", fontsize=9, pad=6, fontweight="bold")
        ax.set_ylabel("Portfolio (indexed)", color="#888888", fontsize=8)
        ax.axhline(100, color="#333344", linewidth=0.7, linestyle=":")

        for r in results_by_regime[regime]:
            if "equity" not in r:
                continue
            eq   = r["equity"]
            eq_n = eq / eq.iloc[0] * 100
            name = r["label"]
            ax.plot(eq_n.index, eq_n.values,
                    color=colors.get(name, "#ffffff"),
                    linestyle=styles.get(name, "-"),
                    linewidth=1.8, alpha=0.9,
                    label=f"{name}\n"
                          f"CAGR {r['cagr']:+.1%} | "
                          f"Sharpe {r['sharpe']:.2f} | "
                          f"Calmar {r['calmar']:.2f}")
        ax.legend(facecolor="#111118", labelcolor="#aaaaaa",
                  fontsize=6.5, framealpha=0.8, loc="upper left")

        # ── Threshold evolution (adaptive strategy only) ───────────────────────
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.set_facecolor("#111118")
        ax2.tick_params(colors="#aaaaaa", labelsize=7)
        ax2.spines[:].set_color("#222233")

        adaptive_r = next((r for r in results_by_regime[regime]
                           if r["label"] == "Adaptive + portfolio confidence"), None)
        if adaptive_r and "thresholds" in adaptive_r and len(adaptive_r["thresholds"]) > 0:
            th = adaptive_r["thresholds"]
            ax2.plot(th.index, th.values, color="#ffaa00", linewidth=0.9, alpha=0.8)
            ax2.axhline(BASE_THRESH, color="#00d4aa", linewidth=0.7,
                        linestyle="--", label=f"Base {BASE_THRESH}")
            ax2.axhline(0.45, color="#ff4444", linewidth=0.7,
                        linestyle=":", label="Max 0.45")
            ax2.set_ylabel("Entry threshold", color="#888888", fontsize=7)
            ax2.set_ylim(0.20, 0.50)
            ax2.legend(facecolor="#111118", labelcolor="#aaaaaa",
                       fontsize=6, framealpha=0.7)
        else:
            ax2.text(0.5, 0.5, "Fixed threshold\n(baseline)",
                     transform=ax2.transAxes, color="#555566",
                     ha="center", va="center", fontsize=8)
            ax2.set_ylabel("Entry threshold", color="#888888", fontsize=7)

    fig.suptitle(
        "Adaptive (baseline) vs Adaptive + Portfolio Confidence  "
        "|  MA100/200 filter  |  ATR stops",
        color="#ffffff", fontsize=11, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Chart → {save_path}")


# ── Print table ────────────────────────────────────────────────────────────────

def print_table(results_by_regime: dict):
    print(f"\n{'═'*100}")
    print(f"  Adaptive (baseline) vs Adaptive + Portfolio Confidence")
    print(f"  Entry threshold rises when portfolio has had a bad recent week")
    print(f"  threshold(t) = 0.25 + max(0, -return_5d)^1.5 × 8.0  clamped to [0.25, 0.45]")
    print(f"  Convex penalty: -1% → +0.008, -3% → +0.042, -6% → +0.117, -8% → +0.181")
    print(f"{'─'*100}")
    print(f"  {'Regime':<28}  {'Strategy':<34}  "
          f"{'Return':>8}  {'CAGR':>7}  {'Sharpe':>7}  "
          f"{'MDD':>7}  {'Calmar':>7}  {'WinRate':>8}  {'Trades':>7}")
    print(f"{'─'*100}")

    order = ["Adaptive (baseline)", "Adaptive + portfolio confidence"]
    for regime, results in results_by_regime.items():
        for label in order:
            r = next((x for x in results if x.get("label")==label), None)
            if r is None:
                continue
            print(f"  {regime:<28}  {label:<34}  "
                  f"{r['return']:>+8.2%}  {r['cagr']:>+7.2%}  "
                  f"{r['sharpe']:>7.2f}  {r['mdd']:>7.2%}  "
                  f"{r['calmar']:>7.2f}  {r['win_rate']:>8.1%}  "
                  f"{r['n_trades']:>7d}")
        print(f"{'─'*100}")
    print(f"{'═'*100}")
    print("""
  Calmar = CAGR / |MDD|  (higher = better return per unit of drawdown)
  Portfolio confidence: threshold rises in proportion to recent portfolio losses.
  Exit threshold is always fixed at -0.25 (confidence only gates entries, not exits).
""")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CAPITAL = 200_000
    results_by_regime = {}

    def _run_regime(data, restrict=None):
        log.info("  Computing scores ...")
        scores = compute_scores(data)
        if not scores:
            return []
        results = []
        for use_pt, label in [
            (False, "Adaptive (baseline)"),
            (True,  "Adaptive + portfolio confidence"),
        ]:
            result = run_backtest(data, scores, capital=CAPITAL,
                                  use_adaptive_threshold=use_pt,
                                  restrict_to_dates=restrict)
            m = metrics(result, label)
            results.append(m)
            log.info(f"    {label}: return={m['return']:+.2%} "
                     f"sharpe={m['sharpe']:.2f} calmar={m['calmar']:.2f} "
                     f"trades={m['n_trades']}")
        return results

    print("\n" + "═"*60)
    print("  Real historical (2021-2025) ...")
    real = load_real(NIFTY_SAMPLE, "2021-01-01", "2025-05-01", warmup_years=4)
    results_by_regime["Real Historical (2021–2025)"] = _run_regime(
        real, restrict=("2021-01-01", "2025-05-01")
    )

    print("\n" + "═"*60)
    print("  Bear market (2019-09 to 2020-06) ...")
    bear = load_real(NIFTY_SAMPLE, "2019-09-01", "2020-06-30", warmup_years=4)
    results_by_regime["Bear Market (2019–2020)"] = _run_regime(
        bear, restrict=("2019-09-01", "2020-06-30")
    )

    print("\n" + "═"*60)
    print("  Synthetic declining ...")
    declining = make_synthetic(n_bars=1500, n_tickers=20,
                               drift=-0.0003, label="dec", seed=42)
    results_by_regime["Declining Synthetic"] = _run_regime(declining)

    print("\n" + "═"*60)
    print("  Random noise ...")
    random_data = make_synthetic(n_bars=1500, n_tickers=20,
                                 drift=0.0, label="rnd", seed=999)
    results_by_regime["Random Noise"] = _run_regime(random_data)

    print_table(results_by_regime)
    plot_results(results_by_regime,
                 RESULTS_DIR / "portfolio_confidence_comparison.png")