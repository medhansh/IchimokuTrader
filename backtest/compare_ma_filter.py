"""
backtest/compare_ma_filter.py
------------------------------
Compares two entry filter approaches:

    A: Hard MA filter (current sheets_trader behaviour)
       Entry allowed only if: close > MA100 AND MA100 > MA200*0.97
       Binary — either passes or doesn't.

    B: Tiered threshold
       No hard MA gate. Instead, the score threshold rises
       proportionally to how far the stock is below its MA100:

           ma_gap       = (close - ma100) / ma100
           adj_threshold = BASE_THRESH + max(0, -ma_gap) * TIER_SENSITIVITY

       Examples (BASE=0.25, SENSITIVITY=1.5):
           close exactly at MA100  → threshold = 0.25  (no penalty)
           close 5% below MA100   → threshold = 0.325
           close 10% below MA100  → threshold = 0.40
           close 20% below MA100  → threshold = 0.55
           close 30% below MA100  → threshold = 0.70  (effectively blocked)

       Stocks in confirmed uptrends get the easy threshold.
       Stocks in moderate pullbacks need a stronger signal.
       Stocks deeply below MA are effectively filtered out anyway.

Four test regimes:
    1. Real historical NSE (2021-2025)
    2. Real bear market (COVID crash 2019-2020)
    3. Synthetic declining market
    4. Synthetic random noise

Run: python3 -m backtest.compare_ma_filter
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.adaptive_signals import adaptive_composite_score

RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
COMMISSION      = 0.001
MAX_POS         = 8
BASE_THRESH          = 0.25    # baseline entry threshold
TIER_SENSITIVITY     = 1.5     # tiered (loose) — 10% below MA100 → +0.15 on threshold
TIER_SENSITIVITY_HIGH= 2.5     # tiered (strict) — 10% below MA100 → +0.25 on threshold
MAX_THRESHOLD        = 0.75    # ceiling — effectively blocks stocks >33%/20% below MA100

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
    "BAJAJ-AUTO", "HDFCAMC", "PIDILITIND", "GODREJCP",
]


# ── Data ───────────────────────────────────────────────────────────────────────

def load_real(tickers, start, end, warmup_years=4):
    import yfinance as yf
    warmup = str(pd.Timestamp(start) - pd.DateOffset(years=warmup_years))[:10]
    data   = {}
    for t in tickers:
        try:
            raw = yf.download(t + ".NS", start=warmup, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()].sort_index().dropna(subset=["Close"])
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
        mv[t] = np.sqrt(omega + ag * mr[t-1]**2 + bg * mv[t-1]**2)
        mr[t] = drift * 0.5 + mv[t] * mi[t]
    for idx in rng_mkt.choice(n_bars, max(1, n_bars // 250 * 5), replace=False):
        mr[idx] += rng_mkt.choice([-1, 1]) * rng_mkt.uniform(0.02, 0.05)

    for i in range(n_tickers):
        rng_t = np.random.default_rng(seed + i * 13 + 7)
        iv = np.zeros(n_bars); iv[0] = vol * 0.8
        ir = np.zeros(n_bars)
        inn = rng_t.standard_t(df=4, size=n_bars) / np.sqrt(2)
        for t in range(1, n_bars):
            iv[t] = np.sqrt(omega + ag * ir[t-1]**2 + bg * iv[t-1]**2)
            ir[t] = drift * 0.5 + iv[t] * inn[t]
        for idx in rng_t.choice(n_bars, rng_t.integers(2, 8), replace=False):
            ir[idx] += rng_t.choice([-1, 1]) * rng_t.uniform(0.03, 0.10)
        beta   = rng_t.uniform(0.5, 1.5)
        tot    = beta * mr + ir
        px     = 500 * np.exp(np.cumsum(tot))
        dr     = np.clip(np.abs(tot) * rng_t.uniform(1.5, 3.0, n_bars), 0.003, 0.08)
        hi     = px * (1 + dr * rng_t.uniform(0.4, 1.0, n_bars))
        lo     = px * (1 - dr * rng_t.uniform(0.4, 1.0, n_bars))
        gap    = rng_t.normal(0, 0.004, n_bars)
        op     = np.roll(px, 1) * (1 + gap); op[0] = px[0]
        vols   = rng_t.lognormal(np.log(1e6), 0.6, n_bars) * (1 + 3 * np.abs(tot) / vol)
        data[f"{label}_{i:03d}"] = pd.DataFrame({
            "Open": op, "High": np.maximum(hi, op),
            "Low": np.minimum(lo, op), "Close": px, "Volume": vols,
        }, index=dates)
    return data


# ── Score computation ──────────────────────────────────────────────────────────

def compute_scores(data: dict) -> dict:
    scores = {}
    for ticker, df in data.items():
        try:
            sc = adaptive_composite_score(df)
            if not sc.dropna().empty:
                scores[ticker] = sc
        except Exception as e:
            log.debug(f"  {ticker}: {e}")
    log.info(f"  Scored {len(scores)}/{len(data)} tickers")
    return scores


# ── MA helpers ─────────────────────────────────────────────────────────────────

def get_ma(data: dict, ticker: str, dt) -> tuple[float, float, float]:
    """Return (close, ma100, ma200) at date dt. Returns (0,0,0) on failure."""
    try:
        hist  = data[ticker][data[ticker].index <= dt]
        c_ser = hist["Close"] if isinstance(hist["Close"], pd.Series) \
                else hist["Close"].iloc[:, 0]
        if len(c_ser) < 200:
            return 0.0, 0.0, 0.0
        return (
            float(c_ser.iloc[-1]),
            float(c_ser.rolling(100).mean().iloc[-1]),
            float(c_ser.rolling(200).mean().iloc[-1]),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def hard_ma_passes(close, ma100, ma200) -> bool:
    """Current sheets_trader filter: binary pass/fail."""
    if ma100 <= 0 or ma200 <= 0:
        return True   # insufficient history — allow through
    return close > ma100 and ma100 > ma200 * 0.97


def tiered_threshold(close, ma100, base=BASE_THRESH,
                     sensitivity=TIER_SENSITIVITY,
                     max_thresh=MAX_THRESHOLD) -> float:
    """
    Compute entry threshold adjusted for distance below MA100.
    The further below MA100, the higher the score needed to enter.
    Stocks above MA100 get the base threshold unchanged.
    """
    if ma100 <= 0:
        return base
    ma_gap = (close - ma100) / ma100      # positive = above MA, negative = below
    penalty = max(0.0, -ma_gap) * sensitivity
    return float(min(base + penalty, max_thresh))


# ── Backtester ─────────────────────────────────────────────────────────────────

def run_backtest(
    data:        dict,
    scores:      dict,
    filter_mode: str,         # "hard_ma" or "tiered"
    capital:     float = 200_000,
    max_pos:     int   = MAX_POS,
    commission:  float = COMMISSION,
    restrict_to_dates=None,
) -> dict:
    """
    filter_mode="hard_ma"  → current behaviour: binary MA100/200 gate
    filter_mode="tiered"   → adaptive threshold based on MA distance
    """
    all_dates = pd.DatetimeIndex(sorted({
        d for df in data.values() for d in df.index.tolist()
    })).sort_values()

    if restrict_to_dates:
        s, e  = pd.Timestamp(restrict_to_dates[0]), pd.Timestamp(restrict_to_dates[1])
        all_dates = all_dates[(all_dates >= s) & (all_dates <= e)]

    if len(all_dates) < 5:
        return {"equity": pd.Series([capital], dtype=float), "trades": []}

    cash, positions, equity, trades = capital, {}, {}, []

    for dt in all_dates:
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

            if close < pos["stop"] or (s_now <= -BASE_THRESH and s_prv > -BASE_THRESH):
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
            open_val = sum(
                pos["shares"] * float(data[t].loc[dt, "Close"])
                for t, pos in positions.items()
                if t in data and dt in data[t].index
            )
            port_val = cash + open_val
            cands    = []

            for t in scores:
                if t in positions or t not in data or dt not in data[t].index:
                    continue
                sc = scores[t]
                if dt not in sc.index:
                    continue

                s_now = float(sc.loc[dt])
                s_prv = float(sc.shift(1).loc[dt]) if dt in sc.index else 0.0
                close, ma100, ma200 = get_ma(data, t, dt)

                if filter_mode == "hard_ma":
                    if not hard_ma_passes(close, ma100, ma200):
                        continue
                    entry_threshold = BASE_THRESH

                elif filter_mode == "tiered":
                    entry_threshold = tiered_threshold(close, ma100,
                                                       sensitivity=TIER_SENSITIVITY)

                elif filter_mode == "tiered_strict":
                    entry_threshold = tiered_threshold(close, ma100,
                                                       sensitivity=TIER_SENSITIVITY_HIGH)

                else:
                    entry_threshold = BASE_THRESH

                # Score crossover check using this entry's threshold
                if not (s_now >= entry_threshold and s_prv < entry_threshold):
                    continue

                cands.append((t, s_now, entry_threshold))

            # Sort by score margin above threshold (highest conviction first)
            cands.sort(key=lambda x: -(x[1] - x[2]))

            for t, s_now, entry_threshold in cands[:max_pos - len(positions)]:
                close, ma100, ma200 = get_ma(data, t, dt)
                if close <= 0:
                    continue

                shares = max(1, math.floor(port_val / max_pos / close))
                cost   = close * shares * (1 + commission)
                if cost > cash:
                    shares = max(0, math.floor(cash * 0.98 / close / (1 + commission)))
                    if shares < 1:
                        continue
                    cost = close * shares * (1 + commission)

                past  = data[t][data[t].index <= dt].tail(20)
                h = past["High"] if isinstance(past["High"], pd.Series) else past["High"].iloc[:, 0]
                l = past["Low"]  if isinstance(past["Low"],  pd.Series) else past["Low"].iloc[:, 0]
                c = past["Close"]if isinstance(past["Close"],pd.Series) else past["Close"].iloc[:, 0]
                tr    = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
                atr_v = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

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

    return {"equity": pd.Series(equity).sort_index(), "trades": trades}


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(result: dict, label: str) -> dict:
    eq     = result["equity"].dropna()
    trades = result["trades"]
    if len(eq) < 2:
        return {"label": label, "return": 0, "cagr": 0, "sharpe": 0,
                "mdd": 0, "calmar": 0, "win_rate": 0, "n_trades": 0}
    ret  = float(eq.iloc[-1] / eq.iloc[0] - 1)
    yrs  = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else 0
    dr   = eq.pct_change().dropna()
    sh   = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    mdd  = float(((eq - eq.cummax()) / eq.cummax()).min())
    pnls = [t["pnl"] for t in trades]
    wr   = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
    cal  = abs(cagr / mdd) if mdd != 0 else 0
    return {"label": label, "return": ret, "cagr": cagr, "sharpe": sh,
            "mdd": mdd, "calmar": cal, "win_rate": wr,
            "n_trades": len(trades), "equity": eq}


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_results(results_by_regime: dict, save_path: Path):
    regimes = list(results_by_regime.keys())
    fig     = plt.figure(figsize=(7 * len(regimes), 6))
    fig.patch.set_facecolor("#0a0a14")
    gs      = gridspec.GridSpec(1, len(regimes), wspace=0.25)

    colors = {
        "Hard MA filter (current)":                       "#00d4aa",
        f"Tiered (sensitivity={TIER_SENSITIVITY})":       "#ffaa00",
        f"Tiered strict (sensitivity={TIER_SENSITIVITY_HIGH})": "#ff6688",
    }
    styles = {
        "Hard MA filter (current)":                       "-",
        f"Tiered (sensitivity={TIER_SENSITIVITY})":       "--",
        f"Tiered strict (sensitivity={TIER_SENSITIVITY_HIGH})": "-.",
    }

    for idx, regime in enumerate(regimes):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#222233")
        ax.set_title(regime, color="#ffffff", fontsize=9, pad=8, fontweight="bold")
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
                          f"Calmar {r['calmar']:.2f} | "
                          f"{r['n_trades']} trades")

        ax.legend(facecolor="#111118", labelcolor="#aaaaaa",
                  fontsize=6.5, framealpha=0.8, loc="upper left")

    fig.suptitle(
        f"MA Filter: Hard Gate vs Tiered 1.5 vs Tiered 2.5  |  "
        f"Tiered: threshold rises as price falls below MA100",
        color="#ffffff", fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Chart → {save_path}")


# ── Print table ────────────────────────────────────────────────────────────────

def print_table(results_by_regime: dict):
    print(f"\n{'═'*102}")
    print(f"  MA FILTER COMPARISON")
    print(f"  Hard MA:         close > MA100 AND MA100 > MA200×0.97  (binary pass/fail)")
    print(f"  Tiered 1.5:      threshold = {BASE_THRESH} + max(0, -ma_gap) × {TIER_SENSITIVITY}  "
          f"(10% below MA100 → thresh {BASE_THRESH + 0.10*TIER_SENSITIVITY:.3f})")
    print(f"  Tiered 2.5:      threshold = {BASE_THRESH} + max(0, -ma_gap) × {TIER_SENSITIVITY_HIGH}  "
          f"(10% below MA100 → thresh {BASE_THRESH + 0.10*TIER_SENSITIVITY_HIGH:.3f})")
    print(f"{'─'*108}")
    print(f"  {'Regime':<28}  {'Filter':<36}  "
          f"{'Return':>8}  {'CAGR':>7}  {'Sharpe':>7}  "
          f"{'MDD':>7}  {'Calmar':>7}  {'WinRate':>8}  {'Trades':>7}")
    print(f"{'─'*108}")

    order = [
        "Hard MA filter (current)",
        f"Tiered (sensitivity={TIER_SENSITIVITY})",
        f"Tiered strict (sensitivity={TIER_SENSITIVITY_HIGH})",
    ]
    for regime, results in results_by_regime.items():
        for label in order:
            r = next((x for x in results if x.get("label") == label), None)
            if r is None:
                continue
            print(f"  {regime:<28}  {label:<36}  "
                  f"{r['return']:>+8.2%}  {r['cagr']:>+7.2%}  "
                  f"{r['sharpe']:>7.2f}  {r['mdd']:>7.2%}  "
                  f"{r['calmar']:>7.2f}  {r['win_rate']:>8.1%}  "
                  f"{r['n_trades']:>7d}")
        print(f"{'─'*108}")

    print(f"{'═'*108}")
    print(f"""
  Calmar = CAGR / |MDD|
  Tiered threshold: no hard MA gate — stocks below MA100 can still enter
  but need a proportionally stronger signal to compensate for the trend risk.
  At 33%+ below MA100 the threshold hits {MAX_THRESHOLD:.2f} (effectively blocked).
""")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CAPITAL = 200_000
    results_by_regime = {}

    def _run_regime(data, label, restrict=None):
        log.info(f"  Computing scores for {label} ...")
        scores = compute_scores(data)
        if not scores:
            return []
        results = []
        for mode, name in [
            ("hard_ma",       "Hard MA filter (current)"),
            ("tiered",        f"Tiered (sensitivity={TIER_SENSITIVITY})"),
            ("tiered_strict", f"Tiered strict (sensitivity={TIER_SENSITIVITY_HIGH})"),
        ]:
            result = run_backtest(data, scores, filter_mode=mode,
                                  capital=CAPITAL, restrict_to_dates=restrict)
            m = metrics(result, name)
            results.append(m)
            log.info(f"    {name}: return={m['return']:+.2%}  "
                     f"sharpe={m['sharpe']:.2f}  calmar={m['calmar']:.2f}  "
                     f"trades={m['n_trades']}")
        return results

    print("\n" + "═"*60)
    print("  Real historical (2021-2025) ...")
    real = load_real(NIFTY_SAMPLE, "2021-01-01", "2025-05-01", warmup_years=4)
    results_by_regime["Real Historical (2021–2025)"] = _run_regime(
        real, "real", restrict=("2021-01-01", "2025-05-01")
    )

    print("\n" + "═"*60)
    print("  Bear market (2019-09 to 2020-06) ...")
    bear = load_real(NIFTY_SAMPLE, "2019-09-01", "2020-06-30", warmup_years=4)
    results_by_regime["Bear Market (2019–2020)"] = _run_regime(
        bear, "bear", restrict=("2019-09-01", "2020-06-30")
    )

    print("\n" + "═"*60)
    print("  Synthetic declining ...")
    declining = make_synthetic(n_bars=1500, n_tickers=20,
                               drift=-0.0003, label="dec", seed=42)
    results_by_regime["Declining Synthetic"] = _run_regime(declining, "declining")

    print("\n" + "═"*60)
    print("  Random noise ...")
    random_data = make_synthetic(n_bars=1500, n_tickers=20,
                                 drift=0.0, label="rnd", seed=999)
    results_by_regime["Random Noise"] = _run_regime(random_data, "random")

    print_table(results_by_regime)
    plot_results(results_by_regime, RESULTS_DIR / "ma_filter_comparison.png")