"""
bot/sheets_trader.py
---------------------
Google Sheets paper trader. Two modes:

    eod  — run after market close (4 PM IST)
           fetches closing prices, computes signals,
           writes pending orders to Pending tab (yellow fill-price column)

    fill — run next morning (9:20 AM IST)
           fetches actual open prices, fills pending orders,
           logs fills to Transactions tab with slippage detail

Auth:
    Set GSHEET_CREDENTIALS env var to the full JSON content of your
    service_account.json. Store as a GitHub Actions secret.
    Set GSHEET_ID to the spreadsheet ID from the sheet URL.

Run:
    python -m bot.sheets_trader --mode eod
    python -m bot.sheets_trader --mode fill
    python -m bot.sheets_trader --mode fill --open-prices "TCS:3920,INFY:1845"
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gspread
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials

from bot.data    import fetch_all_daily, fetch_open_price
from bot.regime  import is_trading_day, market_regime
from bot.tickers import NIFTY500
from strategy.indicators      import atr_latest
from strategy.signals         import entry_signal, exit_signal, SCORE_SELL
from strategy.adaptive_signals import adaptive_latest_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config from env ────────────────────────────────────────────────────────────
SHEET_ID      = os.getenv("GSHEET_ID", "")
SHEET_NAME    = os.getenv("GSHEET_NAME", "ITSP Paper Trades")
MAX_CAPITAL   = float(os.getenv("MAX_CAPITAL",   "200000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS",   "8"))
RISK_PCT      = float(os.getenv("RISK_PER_TRADE","0.02"))
COMMISSION    = 0.001
ATR_TRAIL     = 2.5

# Adaptive strategy uses a lower entry threshold because the WHT multiplier
# scales scores below the original 0.40 level. Calibrated from backtesting.
SCORE_ENTRY   = float(os.getenv("SCORE_THRESHOLD", "0.25"))  # adaptive entry
# Exit threshold stays symmetric but uses original scale (WHT doesn't affect exits
# since we exit on score crossing negative threshold from above)
SCORE_EXIT_THRESH = -SCORE_ENTRY

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ── Tab names ──────────────────────────────────────────────────────────────────
T_HOLDINGS = "Holdings"
T_PENDING  = "Pending"
T_TRANS    = "Transactions"
T_SUMMARY  = "Summary"
T_SIGNALS  = "Signal Radar"    # diagnostic tab — top scores every EOD run

HOLDINGS_HDR = [
    "Ticker", "Entry Date", "Fill Price", "Signal Price",
    "Shares", "Cost (₹)", "Current Price", "Market Value (₹)",
    "Unrealized PnL (₹)", "Return %", "Stop Level", "ATR", "Score", "Status",
]
PENDING_HDR = [
    "Ticker", "Signal Date", "Action", "Signal Price",
    "Suggested Shares", "Est. Value (₹)", "Score", "ATR", "Reason",
    "Fill Price (leave blank for auto-fetch)", "Notes",
]
TRANS_HDR = [
    "Date", "Ticker", "Action", "Signal Price", "Fill Price",
    "Slippage (₹)", "Slippage %", "Shares", "Value (₹)",
    "Commission (₹)", "PnL (₹)", "Return %",
    "Reason", "Cumulative PnL (₹)", "Cumulative Slippage (₹)",
]
SUMMARY_HDR = [
    ["Metric", "Value"],
    ["Initial Capital (₹)",     MAX_CAPITAL],
    ["Current Value (₹)",       ""],
    ["Total Return %",           ""],
    ["Realized PnL (₹)",        ""],
    ["Unrealized PnL (₹)",      ""],
    ["Cumulative Slippage (₹)", ""],
    ["Slippage % of Capital",   ""],
    ["Total Trades",             0],
    ["Win Rate %",               ""],
    ["Avg Win (₹)",              ""],
    ["Avg Loss (₹)",             ""],
    ["Profit Factor",            ""],
    ["Last Updated",             ""],
    ["Market Regime",            ""],
]


# ── Auth ───────────────────────────────────────────────────────────────────────

def get_client() -> gspread.Client:
    creds_json = os.getenv("GSHEET_CREDENTIALS")
    if creds_json:
        creds = Credentials.from_service_account_info(
            json.loads(creds_json), scopes=SCOPES
        )
    else:
        sa = ROOT / "service_account.json"
        if not sa.exists():
            raise FileNotFoundError(
                "No credentials found.\n"
                "Set GSHEET_CREDENTIALS env var or place service_account.json in project root."
            )
        creds = Credentials.from_service_account_file(str(sa), scopes=SCOPES)
    return gspread.authorize(creds)


def open_sheet(gc: gspread.Client) -> gspread.Spreadsheet:
    if SHEET_ID:
        try:
            return gc.open_by_key(SHEET_ID)
        except Exception as e:
            raise RuntimeError(f"Cannot open sheet ID '{SHEET_ID}': {e}\n"
                               "Did you share the sheet with the service account?")
    try:
        return gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        raise RuntimeError(
            f"Sheet '{SHEET_NAME}' not found.\n"
            "Create a blank Google Sheet, share it with the service account, "
            "set GSHEET_ID to its spreadsheet ID, then run --mode setup."
        )


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup(gc: gspread.Client):
    """Initialise tabs and headers in an existing sheet."""
    sh = open_sheet(gc)
    log.info(f"Setting up: {sh.title} ({sh.url})")
    existing = [ws.title for ws in sh.worksheets()]

    def _tab(name, rows=1000, cols=20):
        ws = sh.worksheet(name) if name in existing else sh.add_worksheet(name, rows, cols)
        ws.clear()
        return ws

    wh = _tab(T_HOLDINGS, rows=2000);  wh.update([HOLDINGS_HDR], "A1"); wh.freeze(rows=1)
    wp = _tab(T_PENDING,  rows=500);   wp.update([PENDING_HDR], "A1");  wp.freeze(rows=1)
    wt = _tab(T_TRANS,    rows=5000);  wt.update([TRANS_HDR], "A1");    wt.freeze(rows=1)
    ws = _tab(T_SUMMARY, rows=30, cols=2)
    ws.update(SUMMARY_HDR, "A1")

    # Signal Radar tab
    sr = _tab(T_SIGNALS, rows=120, cols=14)
    sr.update([[
        "As of", "Regime", "Entries", "Stocks above MA100",
        "Near threshold (>80%)", "Ticker", "Score", "Prev Score",
        "Close", "MA100", "MA200", "Eff.Threshold", "Status",
    ]], "A1")
    sr.freeze(rows=1)

    # Remove default Sheet1 if it exists
    try: sh.del_worksheet(sh.worksheet("Sheet1"))
    except Exception: pass

    log.info(f"Setup complete. Share URL: {sh.url}")


# ── Sheet helpers ──────────────────────────────────────────────────────────────

def _read(sh: gspread.Spreadsheet, tab: str) -> pd.DataFrame:
    ws   = sh.worksheet(tab)
    rows = ws.get_all_values()
    if len(rows) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df[df.apply(lambda r: any(v.strip() for v in r), axis=1)]


def _open_tickers(sh: gspread.Spreadsheet) -> set[str]:
    df = _read(sh, T_HOLDINGS)
    if df.empty or "Ticker" not in df.columns:
        return set()
    mask = df.get("Status", pd.Series()) == "OPEN"
    return set(df[mask]["Ticker"].tolist()) if mask.any() else set(df["Ticker"].dropna().tolist())


def _cum_pnl(sh: gspread.Spreadsheet) -> float:
    df = _read(sh, T_TRANS)
    if df.empty or "Cumulative PnL (₹)" not in df.columns:
        return 0.0
    vals = pd.to_numeric(df["Cumulative PnL (₹)"], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else 0.0


def _cum_slip(sh: gspread.Spreadsheet) -> float:
    df = _read(sh, T_TRANS)
    if df.empty or "Cumulative Slippage (₹)" not in df.columns:
        return 0.0
    vals = pd.to_numeric(df["Cumulative Slippage (₹)"], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else 0.0


def _fmt_header(ws: gspread.Worksheet, n: int):
    import gspread.utils as gu
    end = gu.rowcol_to_a1(1, n).rstrip("0123456789")
    ws.format(f"A1:{end}1", {
        "backgroundColor": {"red": 0.07, "green": 0.07, "blue": 0.12},
        "textFormat": {
            "bold": True,
            "foregroundColor": {"red": 0.0, "green": 0.83, "blue": 0.67},
        },
        "horizontalAlignment": "CENTER",
    })


def _colour_row(ws: gspread.Worksheet, row: int, positive: bool):
    # Expand sheet if the row exceeds current grid size
    if row > ws.row_count:
        ws.add_rows(max(100, row - ws.row_count + 50))
    n = ws.col_count
    import gspread.utils as gu
    end = gu.rowcol_to_a1(row, n).rstrip("0123456789")
    bg = {"red": 0.05, "green": 0.17, "blue": 0.12} if positive \
         else {"red": 0.17, "green": 0.05, "blue": 0.08}
    ws.format(f"A{row}:{end}{row}", {"backgroundColor": bg})


# ── EOD MODE ───────────────────────────────────────────────────────────────────

def _trailing_stop(fill_price: float, peak_price: float, atr_v: float) -> float:
    """Return the current trailing stop level based on peak price seen since entry."""
    stop_from_entry = fill_price - ATR_TRAIL * atr_v
    stop_from_peak  = peak_price - ATR_TRAIL * atr_v
    # Stop never moves down — so it's the max of entry-based and peak-based
    return round(max(stop_from_entry, stop_from_peak), 2)


def _write_exit_monitor(
    sh:           gspread.Spreadsheet,
    data:         dict,
    ws:           gspread.Worksheet,
    start_row:    int,
) -> int:
    """
    Append an 'Exit Monitor' section to the Signal Radar tab starting at
    `start_row`.  Shows every open position's:
        - Current price, stop level, distance to stop (%)
        - Current score, distance to score-exit threshold (-0.25)
        - ATR (current, used to compute updated trailing stop)
        - Trailing stop updated from today's high (peak price approximation)
        - Alert status: CRITICAL / WARNING / OK

    Also writes the updated trailing stop back to the Holdings tab.

    Returns the next available row number after writing.
    """
    holdings_df = _read(sh, T_HOLDINGS)
    if holdings_df.empty:
        return start_row

    open_pos = holdings_df[holdings_df.get("Status", pd.Series()) == "OPEN"].copy()
    if open_pos.empty:
        return start_row

    # ── Section header ────────────────────────────────────────────────────────
    ws.update([[
        "EXIT MONITOR — Open Positions",
        "", "", "", "", "", "", "", "", "", "", "", "", "",
    ]], f"A{start_row}", value_input_option="USER_ENTERED")
    ws.format(f"A{start_row}:N{start_row}", {
        "backgroundColor": {"red": 0.10, "green": 0.04, "blue": 0.04},
        "textFormat": {
            "bold": True,
            "fontSize": 11,
            "foregroundColor": {"red": 1.0, "green": 0.4, "blue": 0.4},
        },
    })

    col_header_row = start_row + 1
    ws.update([[
        "Ticker", "Entry Price", "Current Price", "Peak (High)", "ATR",
        "Stop (Stored)", "Stop (Updated)", "Dist to Stop %",
        "Score", "Prev Score", "Score Exit Thr", "Dist to Score Exit",
        "Days Held", "Status",
    ]], f"A{col_header_row}", value_input_option="USER_ENTERED")
    ws.format(f"A{col_header_row}:N{col_header_row}", {
        "backgroundColor": {"red": 0.07, "green": 0.07, "blue": 0.12},
        "textFormat": {
            "bold": True,
            "foregroundColor": {"red": 0.0, "green": 0.83, "blue": 0.67},
        },
        "horizontalAlignment": "CENTER",
    })

    data_start = col_header_row + 1
    exit_rows = []
    holdings_stop_updates = []   # (row_num_in_sheet, new_stop)

    wh = sh.worksheet(T_HOLDINGS)
    all_h_rows = wh.get_all_values()   # for row-number lookup

    for _, pos in open_pos.iterrows():
        ticker = str(pos.get("Ticker", "")).strip().upper()
        if not ticker or ticker not in data:
            continue

        try:
            df_tick = data[ticker]

            # ── Prices ──────────────────────────────────────────────────────
            close_s  = df_tick["Close"] if isinstance(df_tick["Close"], pd.Series) \
                       else df_tick["Close"].iloc[:, 0]
            high_s   = df_tick["High"]  if isinstance(df_tick["High"], pd.Series) \
                       else df_tick["High"].iloc[:, 0]

            current_px = float(close_s.iloc[-1])
            today_high = float(high_s.iloc[-1])

            fill_price = float(pos.get("Fill Price", 0) or 0)
            stored_stop = None
            try:
                stored_stop = float(pos.get("Stop Level", 0) or 0)
            except Exception:
                pass

            # ── ATR (current) ────────────────────────────────────────────────
            atr_v = atr_latest(df_tick)

            # ── Peak price estimate ─────────────────────────────────────────
            # Best proxy: max(today's high, current price, stored_stop + 2.5×ATR)
            # The stored stop already encodes the peak at last EOD:
            #   stored_stop = peak_at_eod - 2.5 × atr_at_eod
            # So: implied_eod_peak = stored_stop + atr_v * ATR_TRAIL
            implied_peak = (stored_stop + atr_v * ATR_TRAIL) if stored_stop else fill_price
            # Today's high may have pushed price above the EOD-implied peak
            peak_price = max(implied_peak, today_high, current_px)

            # ── Updated trailing stop ────────────────────────────────────────
            updated_stop = _trailing_stop(fill_price, peak_price, atr_v)
            # Stored stop acts as a floor (never move stop down between EOD runs)
            if stored_stop and updated_stop < stored_stop:
                updated_stop = stored_stop

            # ── Distance to stop ────────────────────────────────────────────
            dist_to_stop_pct = (current_px - updated_stop) / current_px * 100

            # ── Score ────────────────────────────────────────────────────────
            score_now, score_prev = adaptive_latest_score(df_tick)
            dist_to_score_exit = score_now - SCORE_EXIT_THRESH   # positive = safe margin

            # ── Days held ────────────────────────────────────────────────────
            days_held = ""
            try:
                entry_date = pd.to_datetime(pos.get("Entry Date", ""))
                days_held  = (pd.Timestamp.today() - entry_date).days
            except Exception:
                pass

            # ── Alert status ─────────────────────────────────────────────────
            stop_critical  = dist_to_stop_pct  < 1.0
            stop_warning   = dist_to_stop_pct  < 3.0
            score_critical = dist_to_score_exit < 0.05
            score_warning  = dist_to_score_exit < 0.15

            if stop_critical or score_critical:
                status = "🔴 CRITICAL"
            elif stop_warning or score_warning:
                status = "🟡 WARNING"
            else:
                status = "🟢 OK"

            exit_rows.append({
                "ticker":           ticker,
                "fill_price":       round(fill_price, 2),
                "current_px":       round(current_px, 2),
                "peak_price":       round(peak_price, 2),
                "atr":              round(atr_v, 2),
                "stored_stop":      round(stored_stop, 2) if stored_stop else "",
                "updated_stop":     updated_stop,
                "dist_stop_pct":    round(dist_to_stop_pct, 2),
                "score_now":        round(score_now, 4),
                "score_prev":       round(score_prev, 4),
                "score_exit_thr":   SCORE_EXIT_THRESH,
                "dist_score_exit":  round(dist_to_score_exit, 4),
                "days_held":        days_held,
                "status":           status,
                "stop_critical":    stop_critical,
                "stop_warning":     stop_warning,
                "score_critical":   score_critical,
                "score_warning":    score_warning,
                "updated_stop_raw": updated_stop,
            })

            # ── Queue stop update in Holdings ────────────────────────────────
            for ri, row in enumerate(all_h_rows):
                if ri == 0:
                    continue
                if row[0] == ticker and (len(row) < 14 or row[13] == "OPEN"):
                    # Column K (index 10) = Stop Level  (1-based → col 11)
                    holdings_stop_updates.append((ri + 1, updated_stop))
                    break

        except Exception as e:
            log.debug(f"  ExitMonitor: {ticker} failed: {e}")

    # ── Write exit monitor rows ───────────────────────────────────────────────
    if exit_rows:
        # Sort: CRITICAL first, then WARNING, then OK; within each by dist_stop_pct asc
        priority = {"🔴 CRITICAL": 0, "🟡 WARNING": 1, "🟢 OK": 2}
        exit_rows.sort(key=lambda r: (priority.get(r["status"], 9), r["dist_stop_pct"]))

        sheet_rows = []
        for r in exit_rows:
            sheet_rows.append([
                r["ticker"],
                r["fill_price"],
                r["current_px"],
                r["peak_price"],
                r["atr"],
                r["stored_stop"],
                r["updated_stop"],
                f"{r['dist_stop_pct']:+.2f}%",
                r["score_now"],
                r["score_prev"],
                r["score_exit_thr"],
                r["dist_score_exit"],
                r["days_held"],
                r["status"],
            ])

        ws.update(sheet_rows, f"A{data_start}", value_input_option="USER_ENTERED")

        # Colour rows by alert level
        for i, r in enumerate(exit_rows):
            row_num = data_start + i
            if r["stop_critical"] or r["score_critical"]:
                bg = {"red": 0.25, "green": 0.04, "blue": 0.04}   # deep red
            elif r["stop_warning"] or r["score_warning"]:
                bg = {"red": 0.22, "green": 0.16, "blue": 0.02}   # amber
            else:
                bg = {"red": 0.04, "green": 0.15, "blue": 0.08}   # green
            ws.format(f"A{row_num}:N{row_num}", {"backgroundColor": bg})

        next_row = data_start + len(exit_rows)
    else:
        ws.update([["(no open positions)"]], f"A{data_start}",
                  value_input_option="USER_ENTERED")
        next_row = data_start + 1

    # ── Flush trailing stop updates to Holdings ───────────────────────────────
    if holdings_stop_updates:
        stop_batch = [
            {"range": f"K{row_num}", "values": [[new_stop]]}
            for row_num, new_stop in holdings_stop_updates
        ]
        try:
            wh.batch_update(stop_batch, value_input_option="USER_ENTERED")
            log.info(f"[EXIT MONITOR]  Updated trailing stops for "
                     f"{len(holdings_stop_updates)} positions in Holdings tab")
        except Exception as e:
            log.warning(f"  Could not update trailing stops in Holdings: {e}")

    # Summary log line
    n_crit = sum(1 for r in exit_rows if r["stop_critical"] or r["score_critical"])
    n_warn = sum(1 for r in exit_rows if not (r["stop_critical"] or r["score_critical"])
                 and (r["stop_warning"] or r["score_warning"]))
    log.info(f"[EXIT MONITOR]  {len(exit_rows)} open positions  |  "
             f"{n_crit} CRITICAL  |  {n_warn} WARNING")

    return next_row + 1   # blank separator row


def _write_signal_radar(
    sh:         gspread.Spreadsheet,
    data:       dict,
    regime:     str,
    regime_ok:  bool,
    open_tickers: set,
    top_n:      int = 30,
):
    """
    Write a diagnostic snapshot to the Signal Radar tab.

    Shows the top 30 stocks by score with:
        - Why each was blocked (regime / MA filter / below threshold / no crossover)
        - How close each is to triggering (score vs threshold)
        - MA100/200 status so you can see how many stocks are in uptrends

    This updates every EOD run so you can track market conditions
    over time even when no signals fire.
    """
    rows = []
    for ticker, df in data.items():
        if ticker in open_tickers:
            continue   # already holding — skip
        try:
            score_now, score_prev = adaptive_latest_score(df)
            close_series = df["Close"] if isinstance(df["Close"], pd.Series) \
                           else df["Close"].iloc[:, 0]
            close = float(close_series.iloc[-1])
            ma100 = float(close_series.rolling(100).mean().iloc[-1]) \
                    if len(close_series) >= 100 else 0.0
            ma200 = float(close_series.rolling(200).mean().iloc[-1]) \
                    if len(close_series) >= 200 else 0.0

            # Tiered threshold
            if ma100 > 0:
                ma_gap  = (close - ma100) / ma100
                penalty = max(0.0, -ma_gap) * 2.5
                eff_thr = min(SCORE_ENTRY + penalty, 0.75)
            else:
                eff_thr = SCORE_ENTRY

            crossover = score_now >= eff_thr and score_prev < eff_thr
            pct_to_thr = (eff_thr - score_now) / eff_thr * 100  # % below threshold

            if eff_thr >= 0.75:
                blocked = f"Deeply below MA (thr={eff_thr:.2f})"
            elif not crossover:
                if score_now >= eff_thr:
                    blocked = f"Already above thr={eff_thr:.3f} (no crossover)"
                elif score_now >= eff_thr * 0.80:
                    blocked = f"Near thr={eff_thr:.3f} ({pct_to_thr:.0f}% away)"
                else:
                    blocked = f"Score too low (need {eff_thr:.3f}, have {score_now:.3f})"
            else:
                blocked = "NONE — would fire"

            rows.append({
                "ticker":    ticker,
                "score":     round(score_now, 4),
                "prev":      round(score_prev, 4),
                "close":     round(close, 2),
                "ma100":     round(ma100, 2),
                "ma200":     round(ma200, 2),
                "eff_thr":   round(eff_thr, 3),
                "blocked":   blocked,
                "crossover": crossover,
            })
        except Exception as e:
            log.debug(f"  Radar: {ticker} failed: {e}")

    if not rows:
        return

    # Sort by score descending — highest conviction stocks first
    rows.sort(key=lambda r: -r["score"])
    top = rows[:top_n]

    n_ma_above   = sum(1 for r in rows if r["close"] >= r["ma100"] and r["ma100"] > 0)
    n_near_thresh= sum(1 for r in rows if r["score"] >= r["eff_thr"] * 0.80)

    try:
        ws = sh.worksheet(T_SIGNALS)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(T_SIGNALS, rows=120, cols=14)

    ws.clear()
    # Ensure enough columns for exit monitor (14 columns, A:N)
    if ws.col_count < 14:
        ws.resize(cols=14)

    ws.update([[
        "As of", "Regime", "Entries", "Stocks above MA100",
        "Near threshold (>80%)", "", "", "", "", "", "", "", "",
    ]], "A1", value_input_option="USER_ENTERED")

    ws.update([[
        datetime.now().strftime("%Y-%m-%d %H:%M IST"),
        regime,
        "✓ OK" if regime_ok else "✗ BLOCKED",
        f"{n_ma_above} / {len(rows)}",
        str(n_near_thresh),
        "", "", "", "", "", "", "", "",
    ]], "A2", value_input_option="USER_ENTERED")

    ws.update([[
        "", "", "", "", "",
        "Ticker", "Score", "Prev Score", "Close",
        "MA100", "MA200", "Eff.Threshold", "Status",
    ]], "A3", value_input_option="USER_ENTERED")

    # Format header rows
    ws.format("F3:M3", {
        "backgroundColor": {"red": 0.07, "green": 0.07, "blue": 0.12},
        "textFormat": {"bold": True,
                       "foregroundColor": {"red": 0.0, "green": 0.83, "blue": 0.67}},
        "horizontalAlignment": "CENTER",
    })
    ws.format("A1:E1", {
        "backgroundColor": {"red": 0.07, "green": 0.07, "blue": 0.15},
        "textFormat": {"bold": True,
                       "foregroundColor": {"red": 0.8, "green": 0.8, "blue": 0.9}},
    })

    # Data rows
    data_rows = []
    for r in top:
        data_rows.append([
            "", "", "", "", "",
            r["ticker"],
            r["score"],
            r["prev"],
            r["close"],
            r["ma100"],
            r["ma200"],
            r["eff_thr"],
            r["blocked"],
        ])

    if data_rows:
        ws.append_rows(data_rows, value_input_option="USER_ENTERED")

        # Colour rows by proximity to firing
        for i, r in enumerate(top):
            row_num = i + 4
            if r["score"] >= r["eff_thr"] * 0.90 and r["eff_thr"] < 0.75:
                bg = {"red": 0.05, "green": 0.20, "blue": 0.10}   # green — close to firing
            elif r["eff_thr"] >= 0.75:
                bg = {"red": 0.20, "green": 0.05, "blue": 0.05}   # red — deeply below MA
            else:
                bg = {"red": 0.07, "green": 0.07, "blue": 0.12}   # default
            ws.format(f"A{row_num}:M{row_num}", {"backgroundColor": bg})

    log.info(f"[RADAR]  {n_ma_above}/{len(rows)} stocks above MA100  |  "
             f"{n_near_thresh} near threshold  |  "
             f"Top score: {rows[0]['score']:.3f} ({rows[0]['ticker']}) "
             f"eff_thr={rows[0]['eff_thr']:.3f}")

    # ── Exit Monitor section ─────────────────────────────────────────────
    # Appended below the entry candidates on the same tab.
    # Also writes updated trailing stops back to the Holdings tab.
    exit_section_start = len(top) + 4 + 2   # header rows + data rows + 2 blank separator
    _write_exit_monitor(sh, data, ws, exit_section_start)


def run_eod(sh: gspread.Spreadsheet, force: bool = False):
    log.info(f"EOD MODE  |  {date.today()}")

    ok, reason = is_trading_day()
    if not ok:
        if force:
            log.warning(f"Not a trading day ({reason}) — running anyway due to --force")
        else:
            log.warning(f"Not a trading day: {reason} — skipping EOD run")
            return

    regime_ok, regime = market_regime()
    log.info(f"Regime: {regime}  ({'entries ok' if regime_ok else 'entries blocked'})")

    log.info(f"Fetching {len(NIFTY500)} tickers ...")
    data = fetch_all_daily(NIFTY500)

    open_tickers  = _open_tickers(sh)
    holdings_df   = _read(sh, T_HOLDINGS)
    pending_rows  = []

    for ticker, df in data.items():
        score_now, score_prev = adaptive_latest_score(df)   # ← adaptive FFT+WHT
        close  = float(df["Close"].iloc[-1] if isinstance(df["Close"], pd.Series)
                       else df["Close"].iloc[-1, 0])
        atr_v  = atr_latest(df)
        in_pos = ticker in open_tickers

        if in_pos:
            # Check trailing stop from Holdings sheet
            stop = None
            if not holdings_df.empty and "Stop Level" in holdings_df.columns:
                row = holdings_df[
                    (holdings_df["Ticker"] == ticker) &
                    (holdings_df.get("Status", pd.Series()) == "OPEN")
                ]
                if not row.empty:
                    try: stop = float(row["Stop Level"].iloc[0])
                    except Exception: pass

            stop_hit    = stop is not None and close < stop
            score_exit_ = score_now <= SCORE_EXIT_THRESH and score_prev > SCORE_EXIT_THRESH

            if stop_hit or score_exit_:
                shares = 0
                if not holdings_df.empty and "Shares" in holdings_df.columns:
                    row = holdings_df[holdings_df["Ticker"] == ticker]
                    if not row.empty:
                        try: shares = int(float(row["Shares"].iloc[0]))
                        except Exception: pass
                pending_rows.append([
                    ticker, date.today().isoformat(), "SELL",
                    round(close, 2), shares, round(close * shares, 2),
                    round(score_now, 4), round(atr_v, 2),
                    "trailing_stop" if stop_hit else "score_exit",
                    "", "",
                ])
        else:
            # Entry: adaptive crossover
            if regime_ok and score_now >= SCORE_ENTRY and score_prev < SCORE_ENTRY:
                # ── Tiered threshold filter ────────────────────────────────────
                # No hard MA gate. Instead, threshold rises proportionally
                # as price falls below MA100 (sensitivity=2.5):
                #   At MA100:       threshold = 0.25
                #   5% below MA100: threshold = 0.375
                #   10% below:      threshold = 0.50
                #   20% below:      threshold = 0.75 (effectively blocked)
                try:
                    close_series = df["Close"] if isinstance(df["Close"], pd.Series) \
                                   else df["Close"].iloc[:, 0]
                    if len(close_series) >= 100:
                        ma100 = float(close_series.rolling(100).mean().iloc[-1])
                        ma_gap = (close - ma100) / ma100
                        penalty = max(0.0, -ma_gap) * 2.5
                        effective_threshold = min(SCORE_ENTRY + penalty, 0.75)
                    else:
                        effective_threshold = SCORE_ENTRY
                except Exception:
                    effective_threshold = SCORE_ENTRY

                if score_now >= effective_threshold and score_prev < effective_threshold:
                    slot_capital = MAX_CAPITAL / MAX_POSITIONS
                    shares       = max(1, int(slot_capital / close))
                    stop_level   = round(close - atr_v * ATR_TRAIL, 2)
                    ma_note      = f"thr={effective_threshold:.3f}" \
                                   if effective_threshold > SCORE_ENTRY \
                                   else "at_ma"
                    if shares >= 1:
                        pending_rows.append([
                            ticker, date.today().isoformat(), "BUY",
                            round(close, 2), shares, round(close * shares, 2),
                            round(score_now, 4), round(atr_v, 2),
                            f"score_entry stop={stop_level} {ma_note}",
                            "", "",
                        ])

    # Limit to available slots
    buys  = [r for r in pending_rows if r[2] == "BUY"]
    sells = [r for r in pending_rows if r[2] == "SELL"]
    slots = MAX_POSITIONS - len(open_tickers)
    buys.sort(key=lambda r: -r[6])   # sort by score descending
    pending_rows = sells + buys[:slots]

    # Update current prices for open positions
    _update_prices(sh, data)

    # Write Pending tab — preserve any unfilled orders from previous EOD
    # (can happen if fill run failed or crashed before processing)
    existing_pending = _read(sh, T_PENDING)
    carried_rows = []
    if not existing_pending.empty and "Action" in existing_pending.columns:
        unfilled = existing_pending[existing_pending["Action"].isin(["BUY", "SELL"])]
        if not unfilled.empty:
            log.warning(f"  {len(unfilled)} unfilled orders carried over from previous EOD")
            carried_rows = unfilled.values.tolist()

    wp = sh.worksheet(T_PENDING)
    wp.clear()
    wp.update([PENDING_HDR], "A1")
    _fmt_header(wp, len(PENDING_HDR))

    all_pending = carried_rows + pending_rows
    if all_pending:
        wp.append_rows(all_pending, value_input_option="USER_ENTERED")
        import gspread.utils as gu
        fill_col = PENDING_HDR.index("Fill Price (leave blank for auto-fetch)") + 1
        col_letter = gu.rowcol_to_a1(1, fill_col).rstrip("0123456789")
        wp.format(f"{col_letter}2:{col_letter}{len(all_pending)+1}", {
            "backgroundColor": {"red": 1.0, "green": 0.9, "blue": 0.0},
            "textFormat": {"bold": True,
                           "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}},
        })

    # Write Signal Radar diagnostic tab
    _write_signal_radar(sh, data, regime, regime_ok, open_tickers)

    _update_summary(sh, regime, regime_ok)

    log.info(f"Done: {len(sells)} sell + {len(buys[:slots])} buy orders → Pending tab")
    log.info(f"Sheet: {sh.url}")


def _update_prices(sh: gspread.Spreadsheet, data: dict):
    """Refresh Current Price, Market Value, Unrealized PnL and Return % for open positions."""
    wh  = sh.worksheet(T_HOLDINGS)
    df  = _read(sh, T_HOLDINGS)
    if df.empty:
        return

    # Column indices (1-based) from HOLDINGS_HDR:
    # A=Ticker, B=EntryDate, C=FillPrice, D=SignalPrice, E=Shares,
    # F=Cost(₹), G=CurrentPrice, H=MarketValue, I=UnrealizedPnL, J=Return%
    updates = []
    for i, row in df.iterrows():
        if row.get("Status") != "OPEN":
            continue
        ticker = row["Ticker"]
        if ticker not in data:
            continue
        try:
            current_px = float(data[ticker]["Close"].iloc[-1])
            shares     = int(float(row.get("Shares", 0) or 0))
            cost       = float(row.get("Cost (₹)", 0) or 0)

            market_val = round(current_px * shares, 2)
            unr_pnl    = round(market_val - cost, 2)
            ret_pct    = round(unr_pnl / cost * 100, 2) if cost else 0

            row_num = i + 2   # +1 for 0-index, +1 for header row
            updates.extend([
                {"range": f"G{row_num}", "values": [[current_px]]},
                {"range": f"H{row_num}", "values": [[market_val]]},
                {"range": f"I{row_num}", "values": [[unr_pnl]]},
                {"range": f"J{row_num}", "values": [[ret_pct]]},
            ])
        except Exception as e:
            log.debug(f"  _update_prices: {ticker} failed: {e}")

    if updates:
        wh.batch_update(updates, value_input_option="USER_ENTERED")


# ── FILL MODE ──────────────────────────────────────────────────────────────────

def run_fill(sh: gspread.Spreadsheet, manual_prices_str: str | None = None, force: bool = False):
    log.info(f"FILL MODE  |  {date.today()}")

    ok, reason = is_trading_day()
    if not ok:
        if force:
            log.warning(f"Not a trading day ({reason}) — running anyway due to --force")
        else:
            log.warning(f"Not a trading day: {reason} — pending orders carry over")
            return

    yesterday = date.today() - timedelta(days=1)
    y_ok, y_reason = is_trading_day(yesterday)
    if not y_ok:
        log.info(f"Yesterday ({yesterday}) was not a trading day: {y_reason}")

    manual = {}
    if manual_prices_str:
        for pair in manual_prices_str.split(","):
            try:
                t, p = pair.strip().split(":")
                manual[t.strip().upper()] = float(p.strip())
            except Exception:
                pass

    pending_df = _read(sh, T_PENDING)
    if pending_df.empty or "Ticker" not in pending_df.columns:
        log.info("No pending orders.")
        return

    holdings_df  = _read(sh, T_HOLDINGS)
    wh           = sh.worksheet(T_HOLDINGS)
    wt           = sh.worksheet(T_TRANS)
    cum_pnl      = _cum_pnl(sh)
    cum_slippage = _cum_slip(sh)
    filled       = 0
    slip_today   = 0.0

    for _, order in pending_df.iterrows():
        ticker  = str(order.get("Ticker", "")).strip().upper()
        action  = str(order.get("Action", "")).strip().upper()
        sig_px  = float(order.get("Signal Price", 0) or 0)
        shares  = int(float(order.get("Suggested Shares", 0) or 0))
        score   = float(order.get("Score", 0) or 0)
        atr_v   = float(order.get("ATR", 0) or 0)
        reason  = str(order.get("Reason", ""))
        manual_fill = str(order.get("Fill Price (leave blank for auto-fetch)", "")).strip()

        if not ticker or not action or shares < 1:
            continue

        # Determine fill price
        if manual_fill and manual_fill not in ("", "0"):
            fill_px = float(manual_fill)
            log.info(f"  {ticker}: manual fill ₹{fill_px:.0f}")
        elif ticker in manual:
            fill_px = manual[ticker]
        else:
            log.info(f"  {ticker}: fetching open price ...")
            fill_px = fetch_open_price(ticker)
            if fill_px is None:
                log.warning(f"  {ticker}: no open price — skipping")
                continue
            time.sleep(0.2)

        slippage_rs  = (fill_px - sig_px) * (1 if action == "BUY" else -1) * shares
        slippage_pct = (fill_px / sig_px - 1) * 100 if sig_px else 0
        commission   = fill_px * shares * COMMISSION
        value        = fill_px * shares

        if action == "BUY":
            cost = value + commission
            stop = fill_px - atr_v * ATR_TRAIL
            pnl  = -commission

            wh.append_row([
                ticker, date.today().isoformat(),
                round(fill_px, 2), round(sig_px, 2), shares,
                round(cost, 2), round(fill_px, 2),
                round(value, 2),
                round(value - cost, 2),
                round((value - cost) / cost * 100, 2) if cost else 0,
                round(stop, 2), round(atr_v, 2), round(score, 4), "OPEN",
            ], value_input_option="USER_ENTERED")
            # Get actual last filled row (not grid size)
            last_row = len(wh.get_all_values())
            _colour_row(wh, last_row, positive=True)

        elif action == "SELL":
            entry_cost = value
            if not holdings_df.empty and "Cost (₹)" in holdings_df.columns:
                match = holdings_df[
                    (holdings_df["Ticker"] == ticker) &
                    (holdings_df.get("Status", pd.Series()) == "OPEN")
                ]
                if not match.empty:
                    try: entry_cost = float(match["Cost (₹)"].iloc[0])
                    except Exception: pass

            proceeds = value - commission
            pnl      = proceeds - entry_cost
            ret_pct  = pnl / entry_cost * 100 if entry_cost else 0

            # Mark CLOSED in Holdings
            all_rows = wh.get_all_values()
            for ri, row in enumerate(all_rows):
                if ri == 0: continue
                if row[0] == ticker and (len(row) < 14 or row[13] == "OPEN"):
                    wh.update_cell(ri + 1, 14, "CLOSED")
                    wh.update_cell(ri + 1, 7, round(fill_px, 2))
                    _colour_row(wh, ri + 1, positive=pnl >= 0)
                    break

        cum_pnl      += pnl
        cum_slippage += abs(slippage_rs)
        slip_today   += abs(slippage_rs)

        wt.append_row([
            date.today().isoformat(), ticker, action,
            round(sig_px, 2), round(fill_px, 2),
            round(slippage_rs, 2), f"{slippage_pct:.2f}%",
            shares, round(value, 2),
            round(commission, 2), round(pnl, 2),
            f"{(pnl/value*100):.2f}%" if action == "SELL" else "",
            reason, round(cum_pnl, 2), round(cum_slippage, 2),
        ], value_input_option="USER_ENTERED")
        last_row_t = len(wt.get_all_values())
        _colour_row(wt, last_row_t, positive=pnl >= 0)

        filled += 1
        log.info(f"  ✓ {action} {ticker}: "
                 f"sig=₹{sig_px:.0f} fill=₹{fill_px:.0f} "
                 f"slip={slippage_pct:+.2f}% pnl=₹{pnl:+,.0f}")

    # Clear Pending tab
    wp = sh.worksheet(T_PENDING)
    wp.clear()
    wp.update([PENDING_HDR], "A1")
    _fmt_header(wp, len(PENDING_HDR))

    _, regime = market_regime()
    _update_summary(sh, regime, True)

    log.info(f"\nFilled {filled} orders | "
             f"Cum PnL ₹{cum_pnl:+,.0f} | "
             f"Slip today ₹{slip_today:,.0f} | "
             f"Cum slip ₹{cum_slippage:,.0f} "
             f"({cum_slippage/MAX_CAPITAL*100:.2f}% of capital)")


# ── Summary ────────────────────────────────────────────────────────────────────

def _update_summary(sh: gspread.Spreadsheet, regime: str, regime_ok: bool):
    ws       = sh.worksheet(T_SUMMARY)
    trans_df = _read(sh, T_TRANS)
    hold_df  = _read(sh, T_HOLDINGS)

    n_trades = real_pnl = win_rate = avg_win = avg_loss = pf = cum_slip = 0.0

    if not trans_df.empty and "Action" in trans_df.columns:
        sells = trans_df[trans_df["Action"] == "SELL"].copy()
        sells["PnL (₹)"] = pd.to_numeric(sells["PnL (₹)"], errors="coerce")
        n_trades = len(sells)
        real_pnl = float(sells["PnL (₹)"].sum())
        wins     = sells[sells["PnL (₹)"] > 0]
        losses   = sells[sells["PnL (₹)"] <= 0]
        win_rate = len(wins) / n_trades * 100 if n_trades else 0
        avg_win  = float(wins["PnL (₹)"].mean()) if not wins.empty else 0
        avg_loss = float(losses["PnL (₹)"].mean()) if not losses.empty else 0
        ls       = losses["PnL (₹)"].sum()
        pf       = abs(wins["PnL (₹)"].sum() / ls) if ls != 0 else 0
        if "Cumulative Slippage (₹)" in trans_df.columns:
            v = pd.to_numeric(trans_df["Cumulative Slippage (₹)"], errors="coerce").dropna()
            cum_slip = float(v.iloc[-1]) if not v.empty else 0

    unr = 0.0
    if not hold_df.empty and "Unrealized PnL (₹)" in hold_df.columns:
        open_rows = hold_df[hold_df.get("Status", pd.Series()) == "OPEN"]
        unr = pd.to_numeric(open_rows["Unrealized PnL (₹)"], errors="coerce").sum()

    total_val = MAX_CAPITAL + real_pnl + unr
    total_ret = (total_val / MAX_CAPITAL - 1) * 100

    ws.update([
        [round(total_val, 2)],
        [round(total_ret, 2)],
        [round(real_pnl, 2)],
        [round(unr, 2)],
        [round(cum_slip, 2)],
        [round(cum_slip / MAX_CAPITAL * 100, 3)],
        [int(n_trades)],
        [round(win_rate, 1)],
        [round(avg_win, 2)],
        [round(avg_loss, 2)],
        [round(pf, 2)],
        [datetime.now().strftime("%Y-%m-%d %H:%M IST")],
        [f"{regime} ({'✓' if regime_ok else '✗'})"],
    ], "B2", value_input_option="USER_ENTERED")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["setup", "eod", "fill"], default="eod")
    parser.add_argument("--open-prices", default=None,
                        help="TCS:3920,INFY:1845 — override auto-fetch")
    parser.add_argument("--force", action="store_true",
                        help="Bypass holiday/weekend check (for manual seeding runs)")
    args = parser.parse_args()

    gc = get_client()

    if args.mode == "setup":
        setup(gc)
    else:
        sh = open_sheet(gc)
        if args.mode == "eod":
            run_eod(sh, force=args.force)
        elif args.mode == "fill":
            run_fill(sh, args.open_prices, force=args.force)