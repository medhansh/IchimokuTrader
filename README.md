# IchimokuTrader

Automated paper trading system for NSE equities. Scans the Nifty 500 universe daily, generates signals using an adaptive momentum strategy, and manages a virtual portfolio via Google Sheets — fully automated through GitHub Actions with no broker account required.

Built as part of the ITSP (Institute Technical Summer Project) at IIT Bombay.

---

## How it works

Every trading day, two GitHub Actions jobs run automatically:

**4:00 PM IST — EOD signal scan**
Fetches closing prices for ~330 Nifty 500 stocks, computes adaptive momentum scores, applies regime and trend filters, and writes buy/sell candidates to the **Pending** tab in Google Sheets (highlighted yellow). You review the sheet before anything is committed.

**9:20 AM IST next morning — Fill**
Fetches each stock's actual opening price from yfinance, fills yesterday's pending orders at real prices, and logs the exact slippage (signal price vs fill price) to the **Transactions** tab.

No broker account. No demat. No fees. Just a Google Sheet and a GitHub repo.

---

## Strategy

### Signal computation

Six components are combined into a single composite score in `[-1, 1]`:

```
score = tanh(s1 + s2 + s3 + s4 + s5 + s6) × WHT_multiplier
```

| Component | What it measures |
|---|---|
| s1, s2 | Ichimoku momentum — tenkan vs kijun, with one bar lag |
| s3 | Price position relative to the Senkou cloud |
| s4 | Chikou span confirmation |
| s5 | VIDYA trend — adaptive EMA short vs long |
| s6 | Volume-weighted RSI with dynamic overbought/oversold bounds |

### Adaptive periods (FFT cycle detection)

Unlike standard Ichimoku which uses fixed periods (9/26/52), this implementation detects the **dominant price cycle** per stock per week using FFT on log-prices, then derives all periods from it:

```
tenkan       = cycle × 0.5
kijun        = cycle × 1.0
senkou_b     = cycle × 2.0
displacement = cycle × 0.5
```

This makes the cloud self-calibrating — a stock with a 15-day cycle gets tighter periods than one with a 45-day cycle, automatically.

### WHT regime scaling

A Walsh-Hadamard Transform (WHT) is applied to recent returns to measure market structure. The WHT score acts as a **multiplier** on the composite score (range `[0.7, 1.5]`), not an additive component:

- Structured/trending market → multiplier > 1 (amplify conviction)
- Noisy/random market → multiplier < 1 (dampen uncertainty)

This preserves signal direction while modulating confidence based on market quality.

### Entry conditions (all four must be true)

1. Composite score crosses above **+0.25** (adaptive threshold calibrated from backtesting)
2. Close price > 100-day moving average
3. 100-day MA > 200-day MA × 0.97 (structural uptrend confirmed — rejects bounces in downtrends)
4. Nifty 50 regime is not TRENDING_DOWN (ADX-based regime filter)

### Exit conditions (either)

- Score crosses below **−0.25**
- Price falls below trailing ATR stop: `peak_price − 2.5 × ATR14`

### Position sizing

Each position receives an equal capital slot (`portfolio_value / max_positions`). ATR determines the stop level, not the share count — capital is always fully deployed when signals are available.

---

## Backtest results

Tested on 50 Nifty stocks, 2021–2025, equal slot sizing, ATR trailing stops, MA trend filter:

| Metric | Original strategy | Adaptive FFT+WHT |
|---|---|---|
| Total return | +140% | +326% |
| CAGR | +22.5% | +39.8% |
| Sharpe ratio | 1.50 | 2.32 |
| Max drawdown | −14.6% | −12.0% |
| Win rate | 54% | 60% |

**Out-of-sample stress tests:**

| Regime | Return | Sharpe | Notes |
|---|---|---|---|
| Real bear market (COVID crash 2019–2020) | +3.0% | 1.06 | Nifty fell ~38% in same period |
| Synthetic declining (−7% annual drift) | −0.8% | −0.04 | Near-zero — no long bias |
| Synthetic random noise (zero drift) | +8.6% | 0.87 | Some spurious patterns remain |

The near-zero return on synthetic declining data confirms the strategy is not simply disguised long bias. Synthetic data uses GARCH(1,1) volatility clustering, Student-t fat tails, market factor correlation, and overnight gaps — substantially harder than naive GBM.

---

## Repository structure

```
IchimokuTrader/
│
├── strategy/
│   ├── indicators.py          # Pure math: Ichimoku, VIDYA, VRSI, ATR, ADX
│   ├── signals.py             # Original fixed-period composite score
│   └── adaptive_signals.py   # FFT cycle detection + WHT regime scaling
│
├── bot/
│   ├── tickers.py             # Nifty 500 universe (no imports)
│   ├── data.py                # yfinance EOD and intraday fetching
│   ├── regime.py              # ADX regime filter + NSE holiday calendar
│   └── sheets_trader.py       # Main runner: eod / fill modes
│
├── backtest/
│   ├── vectorbt_runner.py     # Full vectorbt backtest (vectorbt isolated here)
│   └── compare_strategies.py  # Original vs adaptive head-to-head comparison
│
├── tests/
│   └── test_signals.py        # Sanity checks for indicators and signals
│
├── .github/workflows/
│   └── paper_trader.yml       # GitHub Actions: scheduled EOD + fill
│
├── conftest.py                # pytest sys.path fix
├── requirements.txt           # Prod deps: yfinance, gspread, numpy, pandas
└── requirements-dev.txt       # Dev deps: adds vectorbt, matplotlib, pytest
```

**Dependency boundary:** `strategy/` and `bot/` import only numpy, pandas, yfinance, and gspread. vectorbt, scikit-learn, and hmmlearn are confined to `backtest/` and never reach the live bot or GitHub Actions.

---

## Setup

### 1. Google Sheets

1. Create a blank spreadsheet at [sheets.google.com](https://sheets.google.com)
2. Name it `ITSP Paper Trades`
3. Go to [Google Cloud Console](https://console.cloud.google.com) → create a project → enable **Google Sheets API** and **Google Drive API**
4. Create a **Service Account** → download the JSON key
5. Share your spreadsheet with the service account `client_email` (Editor access)
6. Run setup locally:

```bash
pip install -r requirements.txt
export GSHEET_ID="your_spreadsheet_id_from_url"
python -m bot.sheets_trader --mode setup
```

### 2. GitHub Actions

Add three repository secrets under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `GSHEET_CREDENTIALS` | Full contents of your service account JSON file |
| `GSHEET_ID` | Spreadsheet ID from the sheet URL |
| `GSHEET_NAME` | `ITSP Paper Trades` |

Push to `main`. The workflow activates automatically — no cron, no server required.

### 3. First run

Trigger manually from **Actions → Paper Trader → Run workflow** with:
- mode: `eod`
- force: `true` (bypasses weekend check for initial seeding)

---

## Daily workflow

The system is fully automated once set up. Manual intervention is only needed when you want to:

**Override a fill price** — enter the actual open price in the yellow column of the Pending tab before 9:20 AM, or trigger fill manually:

```bash
python -m bot.sheets_trader --mode fill --open-prices "TCS:3920,INFY:1845"
```

**Skip a signal** — delete the row from the Pending tab before 9:20 AM.

**Seed on a weekend** — run EOD manually with `--force` via the Actions UI.

---

## Local development

```bash
# Install dev dependencies (includes vectorbt, matplotlib, pytest)
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run the original strategy backtest
python -m backtest.vectorbt_runner

# Run adaptive vs original comparison across all four test regimes
python -m backtest.compare_strategies

# Run EOD manually
export GSHEET_ID="your_sheet_id"
python -m bot.sheets_trader --mode eod
```

---

## Google Sheet tabs

| Tab | Contents |
|---|---|
| **Holdings** | Open positions with live mark-to-market, stop levels, entry vs current price |
| **Pending** | Orders waiting to fill at next-day open. Yellow column = fill price (leave blank for auto-fetch) |
| **Transactions** | Full audit trail: signal price, fill price, slippage ₹ and %, commission, PnL, cumulative slippage |
| **Summary** | Portfolio value, total return, win rate, profit factor, cumulative slippage as % of capital |

---

## Cost

| Item | Cost |
|---|---|
| GitHub Actions | Free (uses ~600 of 2,000 free minutes/month) |
| Google Sheets API | Free |
| yfinance data | Free (~15 min delay on EOD, fine for swing trading) |
| Broker account | Not required |
| **Total** | **₹0/month** |
