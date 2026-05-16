"""
strategy/indicators.py
-----------------------
Pure indicator math. No side effects, no I/O, no heavy dependencies.
Every function takes a DataFrame and returns a Series or float.

Dependencies: numpy, pandas only.
"""

import numpy as np
import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Safely extract a column as Series regardless of MultiIndex columns."""
    c = df[name]
    return c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c


# ── Ichimoku ───────────────────────────────────────────────────────────────────

def ichimoku(
    df: pd.DataFrame,
    tenkan_period:  int = 32,
    kijun_period:   int = 96,
    senkob_period:  int = 192,
    displacement:   int = 96,
) -> dict[str, pd.Series]:
    """
    Compute full Ichimoku Cloud components.

    Returns dict with keys:
        tenkan, kijun, senkou_a, senkou_b, chikou
    """
    high  = col(df, "High")
    low   = col(df, "Low")
    close = col(df, "Close")

    tenkan  = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun   = (high.rolling(kijun_period).max()  + low.rolling(kijun_period).min())  / 2
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = (
        (high.rolling(senkob_period).max() + low.rolling(senkob_period).min()) / 2
    ).shift(displacement)
    chikou  = close.shift(-displacement)

    return {
        "tenkan":   tenkan,
        "kijun":    kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou":   chikou,
    }


def ichimoku_scores(df: pd.DataFrame, **kwargs) -> tuple[pd.Series, ...]:
    """
    Return the four Ichimoku score components (s1, s2, s3, s4).
    Each is a Series normalised to approximately [-1, 1].
    """
    ic    = ichimoku(df, **kwargs)
    close = col(df, "Close")

    s1 = (ic["tenkan"] - ic["kijun"]) / (ic["kijun"].abs() + 1e-9)
    s2 = s1.shift(1)

    cloud_top = pd.concat([ic["senkou_a"], ic["senkou_b"]], axis=1).max(axis=1)
    s3 = (close - cloud_top) / (cloud_top.abs() + 1e-9)

    s4 = (ic["chikou"] - close) / (close.abs() + 1e-9)

    return s1, s2, s3, s4


# ── VIDYA (Variable Index Dynamic Average) ─────────────────────────────────────

def vidya(
    close: pd.Series,
    sigma_period:    int = 7,
    sigma_ref_short: int = 15,
    sigma_ref_long:  int = 35,
) -> tuple[pd.Series, pd.Series]:
    """
    VIDYA: EMA with adaptive span driven by volatility ratio.

    The recursive formula:
        VIDYA_t = close_t * α*k + VIDYA_{t-1} * (1 - α*k)
    where k = σ / σ_ref and α = 2 / (n+1).

    We approximate with EWM using mean k as effective alpha — fast, vectorised.

    Returns (vidya_short, vidya_long).
    """
    n     = len(close)
    alpha = 2.0 / (n + 1)

    sigma       = close.rolling(sigma_period).std()
    sigma_ref_s = sigma.rolling(sigma_ref_short).mean()
    sigma_ref_l = sigma.rolling(sigma_ref_long).mean()

    k_short = sigma / (sigma_ref_s + 1e-9)
    k_long  = sigma / (sigma_ref_l + 1e-9)

    k_s_mean = float(k_short.mean()) if not k_short.isna().all() else 1.0
    k_l_mean = float(k_long.mean())  if not k_long.isna().all()  else 1.0

    span_s = max(2, int(1.0 / (alpha * k_s_mean + 1e-9)))
    span_l = max(2, int(1.0 / (alpha * k_l_mean + 1e-9)))

    return (
        close.ewm(span=span_s, adjust=False).mean(),
        close.ewm(span=span_l, adjust=False).mean(),
    )


def vidya_score(df: pd.DataFrame) -> pd.Series:
    """VIDYA trend score: (short - long) / long, in [-1, 1] approx."""
    close = col(df, "Close")
    vs, vl = vidya(close)
    return (vs - vl) / (vl.abs() + 1e-9)


# ── Volume-weighted RSI ────────────────────────────────────────────────────────

def volume_weighted_rsi(
    df: pd.DataFrame,
    rsi_period:     int = 21,
    vol_sma_period: int = 30,
) -> pd.Series:
    """
    RSI with dynamic overbought/oversold bounds driven by volume.

    When volume is high relative to its average, the bounds tighten
    (signal becomes more selective). When volume is low, bounds widen.

    Returns a score in [-1, 1]:
        +1  = deeply oversold (strong buy)
        -1  = deeply overbought (strong sell)
         0  = neutral
    """
    close  = col(df, "Close")
    volume = col(df, "Volume")

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=rsi_period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=rsi_period, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    vol_sma   = volume.rolling(vol_sma_period).mean()
    bound_len = (volume / (vol_sma + 1e-9)).apply(np.tanh).abs() * 100
    u_bound   = 100 - bound_len / 2
    d_bound   = bound_len / 2

    score = pd.Series(0.0, index=df.index)
    above = rsi >= u_bound
    below = rsi <= d_bound
    score[above] = -(rsi[above] - u_bound[above]) / (100 - u_bound[above] + 1e-9)
    score[below] =  (d_bound[below] - rsi[below]) / (d_bound[below] + 1e-9)

    return score


# ── ATR ────────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — Wilder EWM smoothing."""
    high  = col(df, "High")
    low   = col(df, "Low")
    close = col(df, "Close")
    prev  = close.shift(1)
    tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def atr_latest(df: pd.DataFrame, period: int = 14) -> float:
    """Return the single latest ATR value."""
    return float(atr(df, period).dropna().iloc[-1])


# ── ADX (for regime detection) ─────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX, DI+, DI-.

    Returns DataFrame with columns: adx, di_plus, di_minus.
    ADX > 25 → trending, < 20 → range-bound.
    """
    high  = col(df, "High")
    low   = col(df, "Low")
    close = col(df, "Close")

    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)

    tr    = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()

    dm_p = pd.Series(
        np.where((high - prev_h) > (prev_l - low), np.maximum(high - prev_h, 0), 0),
        index=df.index,
    )
    dm_m = pd.Series(
        np.where((prev_l - low) > (high - prev_h), np.maximum(prev_l - low, 0), 0),
        index=df.index,
    )

    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / (atr_s + 1e-9)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx_ = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({"adx": adx_, "di_plus": di_p, "di_minus": di_m}, index=df.index)