"""
bot/regime.py
-------------
Market regime detection using ADX.
Self-contained — only numpy, pandas, yfinance.
No sklearn, no hmmlearn.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from strategy.indicators import adx, col

# ── NSE Holiday Calendar 2025 ─────────────────────────────────────────────────
# Update each December with the following year's list from nseindia.com

NSE_HOLIDAYS = {
    date(2025,  2, 26): "Maha Shivratri",
    date(2025,  3, 14): "Holi",
    date(2025,  3, 31): "Id-Ul-Fitr",
    date(2025,  4, 10): "Mahavir Jayanti",
    date(2025,  4, 14): "Ambedkar Jayanti",
    date(2025,  4, 18): "Good Friday",
    date(2025,  5,  1): "Maharashtra Day",
    date(2025,  8, 15): "Independence Day",
    date(2025,  8, 27): "Ganesh Chaturthi",
    date(2025, 10,  2): "Gandhi Jayanti / Dussehra",
    date(2025, 10, 21): "Diwali Laxmi Puja (Muhurat eve session only)",
    date(2025, 10, 22): "Diwali Balipratipada",
    date(2025, 11,  5): "Guru Nanak Jayanti",
    date(2025, 12, 25): "Christmas",
    # 2026 — add when NSE publishes
}

MUHURAT_DAYS = {
    date(2025, 10, 21),   # market open ~6 PM only, not 9:15 AM
}


def is_trading_day(check: date | None = None) -> tuple[bool, str]:
    """
    Returns (True, "") if NSE is open for normal trading.
    Always uses IST (UTC+5:30) — GitHub Actions runs in UTC
    so date.today() would return the wrong date without this.
    """
    if check is None:
        from datetime import timezone, timedelta as td
        ist   = timezone(td(hours=5, minutes=30))
        check = datetime.now(ist).date()

    if check.weekday() >= 5:
        return False, f"Weekend ({check.strftime('%A')})"

    if check in NSE_HOLIDAYS:
        name = NSE_HOLIDAYS[check]
        if check in MUHURAT_DAYS:
            return False, f"Muhurat day — normal session closed ({name})"
        return False, f"NSE holiday: {name}"

    return True, ""


def market_regime() -> tuple[bool, str]:
    """
    Fetch Nifty 50 daily data and return (entries_allowed, regime_label).

    Regime logic (ADX-based):
        ADX > 25 + price > MA50 + DI+ > DI-  → TRENDING_UP    → allow entries
        ADX > 25 + price < MA50 + DI- > DI+  → TRENDING_DOWN  → block entries
        ADX < 20                               → MEAN_REVERTING → allow entries
        20 ≤ ADX ≤ 25                          → TRANSITION     → allow entries
    """
    try:
        raw = yf.download("^NSEI", period="1y", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()].dropna(subset=["Close"])

        adx_df = adx(raw)
        latest = adx_df.dropna().iloc[-1]
        close  = col(raw, "Close")
        ma50   = float(close.rolling(50).mean().iloc[-1])
        price  = float(close.iloc[-1])

        adx_v  = float(latest["adx"])
        dip_v  = float(latest["di_plus"])
        dim_v  = float(latest["di_minus"])

        if adx_v > 25:
            if dim_v > dip_v and price < ma50:
                return False, "TRENDING_DOWN"
            return True, "TRENDING_UP"
        elif adx_v < 20:
            return True, "MEAN_REVERTING"
        else:
            return True, "TRANSITION"

    except Exception as e:
        return True, f"UNKNOWN (err: {e})"