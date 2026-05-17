"""
backtest/monte_carlo.py
------------------------
Monte Carlo simulation on the adaptive strategy's trade outcomes.

Two simulation methods:

    1. Trade shuffling (sequence risk)
       Keeps the exact same set of trade outcomes (same PnL per trade)
       but randomises their order. Shows how sensitive the equity curve
       and drawdown are to the sequence of wins and losses.

    2. Bootstrap resampling (parameter uncertainty)
       Samples trades WITH replacement. Shows the distribution of
       possible Sharpe ratios and returns if the future trade distribution
       matches the past distribution but with sampling variance.

What this tells you:
    - Realistic worst-case drawdown (95th percentile across 10,000 paths)
    - Whether Sharpe 1.93 is statistically significant or could be 1.2
    - Probability of ruin (drawdown exceeding a capital threshold)
    - Whether the portfolio confidence improvement is real signal or noise

Run: python -m backtest.monte_carlo
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SIMS       = 10_000   # number of Monte Carlo paths
RUIN_THRESH  = -0.20    # drawdown at which we consider the strategy "ruined"
CAPITAL      = 200_000


# ── Equity curve from trade list ───────────────────────────────────────────────

def trades_to_equity(
    trade_pnls:    np.ndarray,   # array of PnL values per trade (₹)
    initial:       float = CAPITAL,
    cost_per_trade:float = 0.0,  # already baked into PnL from backtest
) -> np.ndarray:
    """
    Reconstruct equity curve from a sequence of trade PnLs.
    Returns array of portfolio values after each trade.
    """
    equity    = np.empty(len(trade_pnls) + 1)
    equity[0] = initial
    for i, pnl in enumerate(trade_pnls):
        equity[i+1] = equity[i] + pnl
    return equity


def equity_metrics(equity: np.ndarray) -> dict:
    """Compute key metrics from an equity curve (array of portfolio values)."""
    if len(equity) < 2:
        return {"total_return":0,"sharpe":0,"max_drawdown":0,"calmar":0}

    returns      = np.diff(equity) / equity[:-1]
    total_return = equity[-1] / equity[0] - 1
    sharpe       = (returns.mean() / returns.std() * np.sqrt(252 / len(returns))
                    if returns.std() > 0 else 0)
    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    mdd  = float(dd.min())
    # Annualised return (assume ~252/8 = 31.5 bars between trades for 8 positions)
    # Use simple annualisation from total return and number of trades
    n_trades = len(returns)
    years    = n_trades * (252 / 8 / 252)   # rough: 8 avg positions, 252 trading days
    cagr     = (equity[-1]/equity[0])**(1/years) - 1 if years > 0 else total_return
    calmar   = abs(cagr / mdd) if mdd != 0 else 0

    return {
        "total_return": total_return,
        "cagr":         cagr,
        "sharpe":       sharpe,
        "max_drawdown": mdd,
        "calmar":       calmar,
    }


# ── Monte Carlo engines ────────────────────────────────────────────────────────

def simulate_shuffle(
    trade_pnls: np.ndarray,
    n_sims:     int   = N_SIMS,
    initial:    float = CAPITAL,
    seed:       int   = 42,
) -> dict[str, np.ndarray]:
    """
    Method 1: Trade shuffling.

    Keep the exact same trade outcomes, randomise their order.
    This isolates SEQUENCE RISK — how much of your observed Sharpe/drawdown
    was due to lucky vs unlucky ordering.

    Returns dict of metric arrays, one value per simulation.
    """
    rng = np.random.default_rng(seed)
    n   = len(trade_pnls)

    results = {
        "total_return": np.zeros(n_sims),
        "sharpe":       np.zeros(n_sims),
        "max_drawdown": np.zeros(n_sims),
        "calmar":       np.zeros(n_sims),
        "final_equity": np.zeros(n_sims),
    }

    for i in range(n_sims):
        shuffled  = rng.permutation(trade_pnls)
        equity    = trades_to_equity(shuffled, initial)
        m         = equity_metrics(equity)
        results["total_return"][i] = m["total_return"]
        results["sharpe"][i]       = m["sharpe"]
        results["max_drawdown"][i] = m["max_drawdown"]
        results["calmar"][i]       = m["calmar"]
        results["final_equity"][i] = equity[-1]

    return results


def simulate_bootstrap(
    trade_pnls: np.ndarray,
    n_sims:     int   = N_SIMS,
    initial:    float = CAPITAL,
    seed:       int   = 42,
) -> dict[str, np.ndarray]:
    """
    Method 2: Bootstrap resampling with replacement.

    Samples trades WITH replacement — equivalent to asking: if future
    trades come from the same distribution as observed trades but with
    sampling uncertainty, what's the range of outcomes?

    Gives confidence intervals on Sharpe and return estimates.
    """
    rng = np.random.default_rng(seed)
    n   = len(trade_pnls)

    results = {
        "total_return": np.zeros(n_sims),
        "sharpe":       np.zeros(n_sims),
        "max_drawdown": np.zeros(n_sims),
        "calmar":       np.zeros(n_sims),
        "final_equity": np.zeros(n_sims),
    }

    for i in range(n_sims):
        resampled = rng.choice(trade_pnls, size=n, replace=True)
        equity    = trades_to_equity(resampled, initial)
        m         = equity_metrics(equity)
        results["total_return"][i] = m["total_return"]
        results["sharpe"][i]       = m["sharpe"]
        results["max_drawdown"][i] = m["max_drawdown"]
        results["calmar"][i]       = m["calmar"]
        results["final_equity"][i] = equity[-1]

    return results


# ── Summary statistics ─────────────────────────────────────────────────────────

def summarise(
    results:      dict[str, np.ndarray],
    label:        str,
    observed:     dict,
    ruin_thresh:  float = RUIN_THRESH,
) -> dict:
    """Print and return summary statistics from a Monte Carlo run."""

    def pct(arr, p):
        return float(np.percentile(arr, p))

    mdd   = results["max_drawdown"]
    sh    = results["sharpe"]
    ret   = results["total_return"]
    cal   = results["calmar"]

    ruin_rate = (mdd <= ruin_thresh).mean()

    summary = {
        "label":        label,
        "observed":     observed,
        "sharpe_p5":    pct(sh,  5),
        "sharpe_p50":   pct(sh, 50),
        "sharpe_p95":   pct(sh, 95),
        "mdd_p5":       pct(mdd,  5),   # worst 5% (most negative)
        "mdd_p50":      pct(mdd, 50),
        "mdd_p95":      pct(mdd, 95),   # best 5% drawdown (least negative)
        "ret_p5":       pct(ret,  5),
        "ret_p50":      pct(ret, 50),
        "ret_p95":      pct(ret, 95),
        "cal_p50":      pct(cal, 50),
        "ruin_rate":    ruin_rate,
        "results":      results,
    }

    print(f"\n  {'─'*56}")
    print(f"  {label}")
    print(f"  {'─'*56}")
    print(f"  Observed (backtest): "
          f"return={observed['total_return']:+.2%}  "
          f"sharpe={observed['sharpe']:.2f}  "
          f"MDD={observed['max_drawdown']:.2%}")
    print(f"  {'Metric':<20}  {'5th pct':>10}  {'Median':>10}  {'95th pct':>10}")
    print(f"  {'──────':<20}  {'───────':>10}  {'──────':>10}  {'────────':>10}")
    print(f"  {'Total Return':<20}  {pct(ret,5):>+10.2%}  {pct(ret,50):>+10.2%}  {pct(ret,95):>+10.2%}")
    print(f"  {'Sharpe Ratio':<20}  {pct(sh,5):>10.2f}  {pct(sh,50):>10.2f}  {pct(sh,95):>10.2f}")
    print(f"  {'Max Drawdown':<20}  {pct(mdd,5):>10.2%}  {pct(mdd,50):>10.2%}  {pct(mdd,95):>10.2%}")
    print(f"  {'Calmar Ratio':<20}  {pct(cal,5):>10.2f}  {pct(cal,50):>10.2f}  {pct(cal,95):>10.2f}")
    print(f"  Probability of ruin (MDD ≤ {ruin_thresh:.0%}): {ruin_rate:.2%}")

    # Statistical test: is Sharpe significantly > 0?
    t_stat, p_val = stats.ttest_1samp(sh, 0)
    print(f"  H0: Sharpe = 0  →  t={t_stat:.2f}, p={p_val:.4f} "
          f"({'REJECT — edge is real' if p_val < 0.05 else 'FAIL TO REJECT — edge unclear'})")

    return summary


# ── Visualisation ──────────────────────────────────────────────────────────────

def _hist(ax, data, **kwargs):
    """Safe histogram — tries progressively fewer bins on ValueError."""
    for bins in (60, 20, 10, "auto"):
        try:
            ax.hist(data, bins=bins, **kwargs)
            return
        except (ValueError, TypeError):
            continue


def plot_monte_carlo(
    shuffle_results:    dict,
    bootstrap_results:  dict,
    observed_baseline:  dict,
    observed_ptconf:    dict,
    save_path:          Path,
):
    # Guard: skip chart if either strategy has zero-variance results
    for key in ["baseline", "ptconf"]:
        arr = shuffle_results[key]["total_return"]
        if len(arr) == 0 or np.std(arr) == 0:
            log.warning(f"  {key} has zero-variance — skipping chart")
            return
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0a0a14")
    gs  = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30)

    def ax_style(ax, title):
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[:].set_color("#222233")
        ax.set_title(title, color="#ffffff", fontsize=9, pad=6, fontweight="bold")

    # ── Row 0: Shuffle distributions ──────────────────────────────────────────

    # 0,0: Sharpe distribution (shuffle)
    ax = fig.add_subplot(gs[0, 0])
    ax_style(ax, "Sharpe Ratio Distribution\n(trade shuffling)")
    sh_base = shuffle_results["baseline"]["sharpe"]
    sh_pt   = shuffle_results["ptconf"]["sharpe"]
    _hist(ax, sh_base, alpha=0.6, color="#00d4aa",
            label=f"Baseline (obs={observed_baseline['sharpe']:.2f})", density=True)
    _hist(ax, sh_pt, alpha=0.6, color="#ffaa00",
            label=f"Portfolio conf (obs={observed_ptconf['sharpe']:.2f})", density=True)
    ax.axvline(observed_baseline["sharpe"], color="#00d4aa", linewidth=2, linestyle="--")
    ax.axvline(observed_ptconf["sharpe"],   color="#ffaa00", linewidth=2, linestyle="--")
    ax.axvline(0, color="#ff4444", linewidth=1)
    ax.set_xlabel("Sharpe Ratio", color="#aaaaaa", fontsize=8)
    ax.legend(facecolor="#111118", labelcolor="#aaaaaa", fontsize=7)

    # 0,1: Max drawdown distribution (shuffle)
    ax = fig.add_subplot(gs[0, 1])
    ax_style(ax, "Max Drawdown Distribution\n(trade shuffling)")
    mdd_base = shuffle_results["baseline"]["max_drawdown"]
    mdd_pt   = shuffle_results["ptconf"]["max_drawdown"]
    _hist(ax, mdd_base, alpha=0.6, color="#00d4aa",
            label=f"Baseline (obs={observed_baseline['max_drawdown']:.2%})", density=True)
    _hist(ax, mdd_pt, alpha=0.6, color="#ffaa00",
            label=f"Portfolio conf (obs={observed_ptconf['max_drawdown']:.2%})", density=True)
    ax.axvline(observed_baseline["max_drawdown"], color="#00d4aa", linewidth=2, linestyle="--")
    ax.axvline(observed_ptconf["max_drawdown"],   color="#ffaa00", linewidth=2, linestyle="--")
    ax.axvline(RUIN_THRESH, color="#ff4444", linewidth=1.5, label=f"Ruin ({RUIN_THRESH:.0%})")
    ax.set_xlabel("Max Drawdown", color="#aaaaaa", fontsize=8)
    ax.legend(facecolor="#111118", labelcolor="#aaaaaa", fontsize=7)

    # 0,2: Total return distribution (shuffle)
    ax = fig.add_subplot(gs[0, 2])
    ax_style(ax, "Total Return Distribution\n(trade shuffling)")
    ret_base = shuffle_results["baseline"]["total_return"]
    ret_pt   = shuffle_results["ptconf"]["total_return"]
    _hist(ax, ret_base, alpha=0.6, color="#00d4aa",
            label=f"Baseline (obs={observed_baseline['total_return']:+.1%})", density=True)
    _hist(ax, ret_pt, alpha=0.6, color="#ffaa00",
            label=f"Portfolio conf (obs={observed_ptconf['total_return']:+.1%})", density=True)
    ax.axvline(observed_baseline["total_return"], color="#00d4aa", linewidth=2, linestyle="--")
    ax.axvline(observed_ptconf["total_return"],   color="#ffaa00", linewidth=2, linestyle="--")
    ax.axvline(0, color="#ff4444", linewidth=1)
    ax.set_xlabel("Total Return", color="#aaaaaa", fontsize=8)
    ax.legend(facecolor="#111118", labelcolor="#aaaaaa", fontsize=7)

    # ── Row 1: Bootstrap distributions ────────────────────────────────────────

    for col, (key, label, color, obs) in enumerate([
        ("baseline", "Baseline", "#00d4aa", observed_baseline),
        ("ptconf",   "Portfolio Conf", "#ffaa00", observed_ptconf),
    ] + [("diff", "Sharpe Diff (conf - base)", "#ff6688", {})] if False else [
        ("baseline", "Baseline — Bootstrap Sharpe", "#00d4aa", observed_baseline),
        ("ptconf",   "Portfolio Conf — Bootstrap Sharpe", "#ffaa00", observed_ptconf),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax_style(ax, label)
        sh_vals = bootstrap_results[key]["sharpe"]
        p5,p50,p95 = np.percentile(sh_vals,[5,50,95])
        _hist(ax, sh_vals, color=color, alpha=0.7, density=True)
        ax.axvline(obs.get("sharpe",0), color="#ffffff", linewidth=2,
                   linestyle="--", label=f"Observed: {obs.get('sharpe',0):.2f}")
        ax.axvline(p5,  color="#ff4444", linewidth=1, linestyle=":",
                   label=f"5th pct: {p5:.2f}")
        ax.axvline(p95, color="#00ff88", linewidth=1, linestyle=":",
                   label=f"95th pct: {p95:.2f}")
        ax.axvline(0, color="#ff4444", linewidth=0.8)
        ax.set_xlabel("Sharpe Ratio", color="#aaaaaa", fontsize=8)
        ax.set_title(label, color="#ffffff", fontsize=9, pad=6, fontweight="bold")
        ax.legend(facecolor="#111118", labelcolor="#aaaaaa", fontsize=7)

    # 1,2: Bootstrap Sharpe difference
    ax = fig.add_subplot(gs[1, 2])
    ax_style(ax, "Bootstrap Sharpe Difference\n(Portfolio Conf − Baseline)")
    diff = bootstrap_results["ptconf"]["sharpe"] - bootstrap_results["baseline"]["sharpe"]
    _hist(ax, diff, color="#ff6688", alpha=0.7, density=True)
    ax.axvline(0, color="#ff4444", linewidth=2, label="Zero (no difference)")
    ax.axvline(np.percentile(diff, 5),  color="#aaaaaa", linewidth=1,
               linestyle=":", label=f"5th pct: {np.percentile(diff,5):.2f}")
    ax.axvline(np.percentile(diff, 95), color="#aaaaaa", linewidth=1,
               linestyle="-.", label=f"95th pct: {np.percentile(diff,95):.2f}")
    frac_positive = (diff > 0).mean()
    ax.set_xlabel("ΔSharpe", color="#aaaaaa", fontsize=8)
    ax.set_title(
        f"Bootstrap Sharpe Difference\n"
        f"(Portfolio Conf − Baseline)\n"
        f"Prob(conf > base): {frac_positive:.1%}",
        color="#ffffff", fontsize=9, pad=6, fontweight="bold",
    )
    ax.legend(facecolor="#111118", labelcolor="#aaaaaa", fontsize=7)

    fig.suptitle(
        f"Monte Carlo Analysis  |  {N_SIMS:,} simulations  |  "
        f"Adaptive Strategy (NSE, 2021–2025)",
        color="#ffffff", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Chart → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # ── Run the two strategy backtests to get trade lists ─────────────────────
    print("\n" + "═"*60)
    print("  Running backtests to collect trade outcomes ...")

    # Import here to avoid circular at module level
    from backtest.compare_weighting import (
        load_real, compute_scores, run_backtest, NIFTY_SAMPLE
    )

    real = load_real(NIFTY_SAMPLE, "2021-01-01", "2025-05-01", warmup_years=4)
    scores = compute_scores(real)

    restrict = ("2021-01-01", "2025-05-01")
    res_base = run_backtest(real, scores, use_adaptive_threshold=False,
                            restrict_to_dates=restrict)
    res_pt   = run_backtest(real, scores, use_adaptive_threshold=True,
                            restrict_to_dates=restrict)

    pnls_base = np.array([t["pnl"] for t in res_base["trades"]])
    pnls_pt   = np.array([t["pnl"] for t in res_pt["trades"]])

    print(f"  Baseline:          {len(pnls_base)} trades")
    print(f"  Portfolio conf:    {len(pnls_pt)} trades")

    # Observed metrics from actual backtest
    obs_base = equity_metrics(trades_to_equity(pnls_base))
    obs_pt   = equity_metrics(trades_to_equity(pnls_pt))

    # ── Monte Carlo simulations ────────────────────────────────────────────────
    print(f"\n  Running {N_SIMS:,} shuffle simulations ...")
    shuffle_results = {
        "baseline": simulate_shuffle(pnls_base, seed=42),
        "ptconf":   simulate_shuffle(pnls_pt,   seed=43),
    }

    print(f"  Running {N_SIMS:,} bootstrap simulations ...")
    bootstrap_results = {
        "baseline": simulate_bootstrap(pnls_base, seed=44),
        "ptconf":   simulate_bootstrap(pnls_pt,   seed=45),
    }

    # ── Print summaries ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MONTE CARLO RESULTS  |  {N_SIMS:,} simulations")
    print(f"{'═'*60}")

    print(f"\n  ── METHOD 1: Trade Shuffling (sequence risk) ──")
    summarise(shuffle_results["baseline"], "Baseline — Shuffle",
              obs_base, ruin_thresh=RUIN_THRESH)
    summarise(shuffle_results["ptconf"],  "Portfolio Confidence — Shuffle",
              obs_pt,   ruin_thresh=RUIN_THRESH)

    print(f"\n  ── METHOD 2: Bootstrap Resampling (parameter uncertainty) ──")
    summarise(bootstrap_results["baseline"], "Baseline — Bootstrap",
              obs_base, ruin_thresh=RUIN_THRESH)
    summarise(bootstrap_results["ptconf"],  "Portfolio Confidence — Bootstrap",
              obs_pt,   ruin_thresh=RUIN_THRESH)

    # ── Key question: is portfolio confidence better? ─────────────────────────
    diff = bootstrap_results["ptconf"]["sharpe"] - bootstrap_results["baseline"]["sharpe"]
    prob_better = (diff > 0).mean()
    print(f"\n{'═'*60}")
    print(f"  KEY QUESTION: Is portfolio confidence strategy better?")
    print(f"  P(conf Sharpe > baseline Sharpe) = {prob_better:.1%}")
    if prob_better > 0.75:
        print(f"  → Confidence mechanism likely adds value")
    elif prob_better > 0.50:
        print(f"  → Weak evidence — more data needed")
    else:
        print(f"  → Baseline is likely better — confidence mechanism not adding value")
    print(f"{'═'*60}")

    # ── Chart ──────────────────────────────────────────────────────────────────
    plot_monte_carlo(
        shuffle_results, bootstrap_results,
        obs_base, obs_pt,
        RESULTS_DIR / "monte_carlo.png",
    )