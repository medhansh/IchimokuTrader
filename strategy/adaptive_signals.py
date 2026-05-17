"""
strategy/adaptive_signals.py
-----------------------------
Adaptive Ichimoku + VIDYA + VRSI with FFT cycle detection and WHT regime score.

Key design changes vs v1:
    1. Correct Ichimoku period ratios
       Original Ichimoku uses 9:26:52:26 (tenkan:kijun:senkouB:displacement).
       Ratio is approximately 1 : 2.9 : 5.8 : 2.9.
       We preserve this ratio scaled to the detected cycle:
           tenkan       = cycle // 2            (half-cycle)
           kijun        = cycle                 (full cycle — the anchor)
           senkou_b     = cycle * 2             (double cycle)
           displacement = cycle // 2            (half-cycle, NOT full cycle)
       Note: displacement = kijun/2 keeps the cloud from projecting too far
       forward, which was the main cause of too-few entries in v1.

    2. WHT as MULTIPLIER not additive component
       Instead of adding WHT score to the composite (which penalised every
       bar during noisy periods), we use WHT to scale the composite score:
           final = tanh(raw) * wht_multiplier
       where wht_multiplier ∈ [0.3, 1.3]:
           structured market  → multiplier > 1  (amplify confident signals)
           random/noisy market → multiplier < 1  (dampen uncertain signals)
       This preserves signal direction while modulating conviction.

    3. FFT uses log-prices not raw prices
       Log prices better satisfy the stationarity assumption of FFT.
       Detrending log prices = removing the compound growth component.

    4. Smoother cycle transitions
       Use exponential smoothing on the detected cycle series to prevent
       abrupt period jumps that cause whipsawing in the Ichimoku signals.

Dependencies: numpy, pandas only.
"""

import numpy as np
import pandas as pd

from strategy.indicators import col, vidya_score, volume_weighted_rsi


# ── Configuration ──────────────────────────────────────────────────────────────

FFT_WINDOW   = 128    # FFT lookback (power of 2)
CYCLE_MIN    = 10     # minimum cycle period (bars)
CYCLE_MAX    = 60     # maximum cycle period (bars)
FFT_TOP_N    = 6      # frequencies to keep in noise-filtered price
CYCLE_STEP   = 10     # recompute cycle every N bars
CYCLE_SMOOTH = 0.15   # EMA smoothing factor for cycle series (lower = smoother)

WHT_WINDOW   = 128    # WHT lookback — 128 days ≈ 6 months, long enough to
                      # distinguish genuine market cycles from GARCH vol clustering
WHT_AMP_MAX  = 1.5
WHT_AMP_MIN  = 0.7

# Ichimoku period ratios relative to detected cycle
# Original: tenkan=9, kijun=26, senkouB=52, displacement=26
# Ratios:   0.35,     1.0,      2.0,         1.0
# We use a slightly tighter displacement (0.5 instead of 1.0) to generate
# more signals while preserving the cloud's predictive structure.
TENKAN_RATIO  = 0.5    # tenkan  = cycle * 0.5
KIJUN_RATIO   = 1.0    # kijun   = cycle * 1.0
SENKOB_RATIO  = 2.0    # senkouB = cycle * 2.0
DISPLACE_RATIO= 0.5    # displacement = cycle * 0.5  ← key fix vs v1


# ── FFT cycle detection ────────────────────────────────────────────────────────

def dominant_cycle(log_prices: np.ndarray, window: int = FFT_WINDOW) -> int:
    """
    Detect dominant cycle from log-price series using FFT.

    Steps:
        1. Take last `window` log-prices
        2. Remove linear trend (log drift)
        3. Apply Hanning window (reduce spectral leakage)
        4. FFT → power spectrum
        5. Find peak in valid frequency range
        6. Return period in bars

    Using log prices is critical: raw prices have non-stationary variance
    (price levels change) while log-returns are more stationary.
    """
    if len(log_prices) < window:
        return 20   # sensible default

    seg   = log_prices[-window:].astype(float)
    x     = np.arange(window)
    slope = np.polyfit(x, seg, 1)
    seg   = seg - np.polyval(slope, x)          # detrend
    seg  *= np.hanning(window)                   # reduce leakage

    fft   = np.fft.rfft(seg)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(window)

    # Valid range: cycles between CYCLE_MIN and CYCLE_MAX bars
    mask  = (freqs > 1.0/CYCLE_MAX) & (freqs < 1.0/CYCLE_MIN) & (freqs > 0)
    if not mask.any():
        return 20

    dom_freq = freqs[mask][np.argmax(power[mask])]
    period   = int(round(1.0 / dom_freq))
    return int(np.clip(period, CYCLE_MIN, CYCLE_MAX))


def cycle_series(close: pd.Series) -> pd.Series:
    """
    Rolling FFT cycle detection with exponential smoothing.

    Smoothing prevents abrupt period jumps that cause indicator whipsawing.
    The detected cycle evolves slowly, adapting over weeks not days.
    """
    log_prices = np.log(close.values.astype(float) + 1e-9)
    n          = len(log_prices)
    raw_cycles = np.full(n, 20.0)  # default

    for i in range(FFT_WINDOW, n, CYCLE_STEP):
        raw_cycles[i] = float(dominant_cycle(log_prices[:i]))

    # Forward fill between computation points
    for i in range(1, n):
        if raw_cycles[i] == 20.0 and i > 0:
            raw_cycles[i] = raw_cycles[i-1]

    # Exponential smoothing to avoid abrupt jumps
    smoothed = np.zeros(n)
    smoothed[0] = raw_cycles[0]
    for i in range(1, n):
        smoothed[i] = CYCLE_SMOOTH * raw_cycles[i] + (1 - CYCLE_SMOOTH) * smoothed[i-1]

    cycles = np.round(smoothed).astype(int)
    cycles = np.clip(cycles, CYCLE_MIN, CYCLE_MAX)
    return pd.Series(cycles, index=close.index)


# ── FFT noise filter ───────────────────────────────────────────────────────────

def fft_filtered_close(close: pd.Series, top_n: int = FFT_TOP_N) -> pd.Series:
    """
    Reconstruct close price keeping only top_n frequency components.
    Operates on log prices to handle non-stationarity, then exponentiates back.
    """
    log_p   = np.log(close.values.astype(float) + 1e-9)
    fft     = np.fft.rfft(log_p)
    power   = np.abs(fft) ** 2
    top_idx = np.argsort(power)[-top_n:]
    filtered= np.zeros_like(fft, dtype=complex)
    filtered[top_idx] = fft[top_idx]
    reconstructed = np.fft.irfft(filtered, n=len(log_p))
    return pd.Series(np.exp(reconstructed), index=close.index)


def build_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Close/High/Low with FFT-filtered prices."""
    close  = col(df, "Close")
    high   = col(df, "High")
    low    = col(df, "Low")

    f_close = fft_filtered_close(close)
    hl_half = (high - low) / 2
    f_high  = f_close + hl_half
    f_low   = f_close - hl_half

    out = df.copy()
    out["Close"] = f_close.values
    out["High"]  = f_high.values
    out["Low"]   = f_low.values
    return out


# ── Walsh-Hadamard Transform ───────────────────────────────────────────────────

def wht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (in-place, iterative)."""
    n    = len(x)
    assert (n & (n-1)) == 0, "WHT requires power-of-2 length"
    h    = x.astype(float).copy()
    step = 1
    while step < n:
        for i in range(0, n, step*2):
            for j in range(i, i+step):
                a, b         = h[j], h[j+step]
                h[j]         = a + b
                h[j+step]    = a - b
        step *= 2
    return h / n


def wht_multiplier(returns: np.ndarray, window: int = WHT_WINDOW) -> float:
    """
    Compute a score multiplier from WHT energy concentration.

    Structured/trending market → concentration high → multiplier > 1
    Random/noisy market        → concentration low  → multiplier < 1

    Baseline: white noise has uniform WHT energy → concentration ≈ 1/window
    A pure trend has energy concentrated in first few coefficients.

    Returns multiplier ∈ [WHT_AMP_MIN, WHT_AMP_MAX].
    """
    if len(returns) < window:
        return 1.0

    seg = returns[-window:].astype(float)
    seg = seg - seg.mean()

    if np.std(seg) < 1e-10:
        return 1.0

    h      = wht(seg)
    energy = h ** 2
    total  = energy.sum()
    if total < 1e-10:
        return 1.0

    # Energy in first 12.5% of coefficients (most structured = low sequency)
    low_n       = max(1, window // 8)
    low_energy  = energy[:low_n].sum()
    concentration = low_energy / total

    # White noise baseline: ~1/8 of energy in first 1/8 of coefficients
    baseline    = 1.0 / 8.0
    # Scale: baseline → 1.0, fully concentrated → WHT_AMP_MAX, fully diffuse → WHT_AMP_MIN
    t           = (concentration - baseline) / (1.0 - baseline)  # [0, 1]
    t           = np.clip(t, 0.0, 1.0)
    multiplier  = WHT_AMP_MIN + t * (WHT_AMP_MAX - WHT_AMP_MIN)
    return float(multiplier)


def wht_multiplier_series(close: pd.Series) -> pd.Series:
    """Rolling WHT multiplier series."""
    returns = close.pct_change().fillna(0).values
    mults   = np.ones(len(close))
    for i in range(WHT_WINDOW, len(returns)):
        mults[i] = wht_multiplier(returns[i-WHT_WINDOW:i])
    return pd.Series(mults, index=close.index)


# ── Adaptive Ichimoku ──────────────────────────────────────────────────────────

def _ichimoku_scores_for_period(
    high: pd.Series, low: pd.Series, close: pd.Series,
    t_per: int, k_per: int, sb_per: int, disp: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute the four Ichimoku score components for given periods."""
    tenkan   = (high.rolling(t_per).max()  + low.rolling(t_per).min())  / 2
    kijun    = (high.rolling(k_per).max()  + low.rolling(k_per).min())  / 2
    senkA    = ((tenkan + kijun) / 2).shift(disp)
    senkB    = ((high.rolling(sb_per).max() + low.rolling(sb_per).min()) / 2).shift(disp)
    chikou   = close.shift(-disp)
    cloud_top= pd.concat([senkA, senkB], axis=1).max(axis=1)

    s1 = (tenkan - kijun)    / (kijun.abs()     + 1e-9)
    s2 = s1.shift(1)
    s3 = (close - cloud_top) / (cloud_top.abs() + 1e-9)
    s4 = (chikou - close)    / (close.abs()      + 1e-9)
    return s1, s2, s3, s4


def adaptive_ichimoku_scores(
    df:       pd.DataFrame,
    cycle_s:  pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Ichimoku scores with periods derived from detected cycle.

    Period derivation (preserving original 1:2.9:5.8 Ichimoku ratio):
        tenkan       = max(5,  round(cycle * 0.5))
        kijun        = max(10, round(cycle * 1.0))
        senkou_b     = max(20, round(cycle * 2.0))
        displacement = max(5,  round(cycle * 0.5))

    Computed separately for each unique cycle value detected,
    then stitched together by date mask.
    """
    high  = col(df, "High")
    low   = col(df, "Low")
    close = col(df, "Close")

    s1_out = pd.Series(np.nan, index=df.index)
    s2_out = pd.Series(np.nan, index=df.index)
    s3_out = pd.Series(np.nan, index=df.index)
    s4_out = pd.Series(np.nan, index=df.index)

    # Group bars by cycle value and compute once per unique value
    unique_cycles = sorted(cycle_s.unique())
    for cyc in unique_cycles:
        t_per  = max(5,  int(round(cyc * TENKAN_RATIO)))
        k_per  = max(10, int(round(cyc * KIJUN_RATIO)))
        sb_per = max(20, int(round(cyc * SENKOB_RATIO)))
        disp   = max(5,  int(round(cyc * DISPLACE_RATIO)))

        s1, s2, s3, s4 = _ichimoku_scores_for_period(
            high, low, close, t_per, k_per, sb_per, disp
        )
        mask = cycle_s == cyc
        s1_out[mask] = s1[mask]
        s2_out[mask] = s2[mask]
        s3_out[mask] = s3[mask]
        s4_out[mask] = s4[mask]

    return s1_out, s2_out, s3_out, s4_out


# ── Composite adaptive score ───────────────────────────────────────────────────

def adaptive_composite_score(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive composite score: Ichimoku(adaptive) + VIDYA + VRSI, scaled by WHT.

    Formula:
        raw     = s1 + s2 + s3 + s4 + s5 + s6
        base    = tanh(raw)
        final   = base * wht_multiplier   (WHT scales conviction, never flips sign)

    Min bars: FFT_WINDOW + WHT_WINDOW + CYCLE_MAX*2 + 50 for adequate warmup.
    Returns empty Series only if truly insufficient data.
    """
    # Conservative minimum: enough for FFT + WHT + largest possible Ichimoku window
    min_needed = FFT_WINDOW + WHT_WINDOW + int(CYCLE_MAX * SENKOB_RATIO) + 50
    if len(df) < min_needed:
        return pd.Series(dtype=float)

    close = col(df, "Close")

    # 1. FFT-filtered price
    df_f  = build_filtered_df(df)

    # 2. Adaptive cycle detection (smoothed)
    cyc_s = cycle_series(close)

    # 3. Adaptive Ichimoku on filtered price
    s1, s2, s3, s4 = adaptive_ichimoku_scores(df_f, cyc_s)

    # 4. VIDYA and VRSI (unchanged from original)
    s5 = vidya_score(df)
    s6 = volume_weighted_rsi(df)

    # 5. Raw composite and base score
    raw  = s1 + s2 + s3 + s4 + s5 + s6
    base = raw.apply(np.tanh)

    # 6. WHT multiplier
    wht_mult = wht_multiplier_series(close)

    # 7. Final score clipped to [-1, 1]
    final = (base * wht_mult).clip(-1, 1)

    # Drop leading NaNs but return full-length series aligned to df.index
    return final


def adaptive_latest_score(df: pd.DataFrame) -> tuple[float, float]:
    """Return (score_now, score_prev) for live trading."""
    sc = adaptive_composite_score(df).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])