"""
bot/data.py
-----------
yfinance data fetching for the live bot.
No heavy dependencies.
"""

import time
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and deduplicate."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.loc[:, ~raw.columns.duplicated()].sort_index()


def fetch_daily(ticker: str, min_bars: int = 400) -> Optional[pd.DataFrame]:
    """
    Fetch 4 years of daily OHLCV for one NSE ticker.
    Returns None if insufficient data or fetch fails.
    """
    try:
        raw = yf.download(
            ticker + ".NS", period="4y", interval="1d",
            auto_adjust=True, progress=False, timeout=15,
        )
        df = _clean(raw).dropna(subset=["Close"])
        return df if len(df) >= min_bars else None
    except Exception:
        return None


def fetch_open_price(ticker: str) -> Optional[float]:
    """
    Fetch today's actual open price from 1-min bars.
    Returns None if market hasn't opened yet or fetch fails.
    """
    try:
        raw = yf.download(
            ticker + ".NS", period="1d", interval="1m",
            auto_adjust=True, progress=False, timeout=10,
        )
        df    = _clean(raw).sort_index()
        today = df[df.index.date == date.today()]
        return float(today["Open"].iloc[0]) if not today.empty else None
    except Exception:
        return None


def fetch_all_daily(
    tickers: list[str],
    min_bars: int = 400,
    delay: float = 0.08,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily data for all tickers. Logs progress every 50.
    Returns dict of {ticker: DataFrame} for tickers with sufficient data.
    """
    import logging
    log = logging.getLogger(__name__)

    data, failed = {}, []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        df = fetch_daily(ticker, min_bars)
        if df is not None:
            data[ticker] = df
        else:
            failed.append(ticker)
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{total} fetched ({len(data)} ok, {len(failed)} failed)")
        time.sleep(delay)

    log.info(f"Fetch complete: {len(data)}/{total} tickers loaded")
    return data