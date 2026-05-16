# IchimokuTrader

Automated paper trading system for NSE equities using Ichimoku + VIDYA + Volume-weighted RSI.

## Structure

```
IchimokuTrader/
├── strategy/
│   ├── indicators.py   # Ichimoku, VIDYA, VRSI, ATR, ADX — pure math
│   └── signals.py      # composite score, entry/exit, position sizing
├── bot/
│   ├── tickers.py      # Nifty 500 universe
│   ├── data.py         # yfinance data fetching
│   ├── regime.py       # ADX regime filter + NSE holiday calendar
│   └── sheets_trader.py # main runner (eod / fill)
├── backtest/
│   └── vectorbt_runner.py  # backtesting — vectorbt lives here only
├── tests/
│   └── test_signals.py
├── requirements.txt        # prod deps (no vectorbt)
└── requirements-dev.txt    # dev deps (vectorbt, streamlit, etc.)
```

## Quick start

```bash
# Clone and install prod deps
pip install -r requirements.txt

# Place your service_account.json in the project root
# Create a blank Google Sheet, share it with the service account email
# Set your sheet ID:
export GSHEET_ID="your_spreadsheet_id"

# First-time setup (creates tabs and headers)
python -m bot.sheets_trader --mode setup

# Run EOD signals (after 3:30 PM IST)
python -m bot.sheets_trader --mode eod

# Fill pending orders next morning (after 9:15 AM IST)
python -m bot.sheets_trader --mode fill

# Or with manual open prices
python -m bot.sheets_trader --mode fill --open-prices "TCS:3920,INFY:1845"
```

## Backtesting (local only)

```bash
pip install -r requirements-dev.txt
python -m backtest.vectorbt_runner
```

## Tests

```bash
pytest tests/ -v
```

## GitHub Actions

Two scheduled runs daily (IST):
- **9:20 AM** → fill pending orders at actual open prices
- **4:00 PM** → generate EOD signals

**Secrets required:**
- `GSHEET_CREDENTIALS` — full JSON content of service_account.json
- `GSHEET_ID` — spreadsheet ID from the sheet URL
- `GSHEET_NAME` — sheet name (default: "ITSP Paper Trades")

## Strategy

**Signal:** tanh(s1 + s2 + s3 + s4 + s5 + s6) where:
- s1, s2 = Ichimoku momentum (tenkan vs kijun, lagged)
- s3 = price position relative to cloud
- s4 = chikou confirmation
- s5 = VIDYA trend (adaptive EMA short vs long)
- s6 = volume-weighted RSI

**Entry:** score crosses above +0.40  
**Exit:** score crosses below -0.40 OR trailing ATR stop hit  
**Sizing:** ATR risk sizing × score-proportional scaling  
**Regime filter:** blocks new entries when Nifty ADX > 25 and trending down

**Backtest results (2021–2025, Nifty 500):**
- Total return: +251% vs Nifty +73%
- Sharpe ratio: 2.30
- Max drawdown: -13.82%
- Win rate: 51%