"""
strategy/gmm_weighted_signals.py
----------------------------------
Student-t Mixture Model (TMM) with soft assignments for score weighting.

Why Student-t instead of Gaussian:
    Stock market score components exhibit fat tails — extreme values
    occur far more often than a Gaussian predicts. Using Gaussian
    mixture components causes two problems:
        1. Extreme observations get near-zero likelihood under all components
           → responsibilities become numerically unstable
           → weight matrix converges to near-uniform (observed empirically)
        2. The model "protects" itself from outliers by shrinking component
           variances, which makes the Gaussians too tight and miss the tails

    Student-t distribution with ν degrees of freedom has tails that decay
    as |x|^{-(ν+1)} instead of exp(-x²). For ν=4 (typical NSE stocks),
    this gives ~16× more probability mass at 3σ than a Gaussian.

EM Algorithm for Student-t Mixture:
    The Student-t can be written as a Gaussian scale mixture:
        x | u ~ N(μ, Σ/u)
        u ~ Gamma(ν/2, ν/2)
    This allows a clean EM formulation with closed-form updates.

    E-step:
        Responsibility: r_{tk} = π_k × t(x_t; μ_k, Σ_k, ν_k) / Σ_k [...]
        Weight:         w_{tk} = (ν_k + d) / (ν_k + δ_{tk})
        where δ_{tk} = (x_t - μ_k)ᵀ Σ_k⁻¹ (x_t - μ_k)  [Mahalanobis dist²]

    M-step:
        π_k = Σ_t r_{tk} / T
        μ_k = Σ_t (r_{tk} w_{tk} x_t) / Σ_t (r_{tk} w_{tk})
        Σ_k = Σ_t (r_{tk} w_{tk} (x_t-μ_k)(x_t-μ_k)ᵀ) / Σ_t r_{tk}
        ν_k: solve via Newton's method on the log-likelihood

    The w_{tk} weight acts as a per-observation robustness weight —
    outliers get small w (down-weighted) rather than disrupting μ and Σ.

Dependencies: numpy, scipy, pandas — no sklearn needed.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from strategy.indicators import col
from strategy.adaptive_signals import (
    cycle_series, build_filtered_df, adaptive_ichimoku_scores,
    wht_multiplier_series, WHT_WINDOW,
)
from strategy.signals import vidya_score, volume_weighted_rsi, MIN_BARS_NEEDED

log = logging.getLogger(__name__)

MODEL_PATH     = Path(__file__).resolve().parent.parent / "data" / "tmm_model.pkl"
N_COMPONENTS   = 3      # number of mixture components
NU_INIT        = 4.0    # initial degrees of freedom (4 = realistic for equity)
NU_MIN         = 2.1    # minimum ν (must be > 2 for finite variance)
NU_MAX         = 30.0   # maximum ν (above ~30, t ≈ Gaussian)
MIN_TRAIN_BARS = MIN_BARS_NEEDED + 50
N_SCORE_COMP   = 6      # s1 through s6


# ── Score component extraction ─────────────────────────────────────────────────

def extract_components(df: pd.DataFrame) -> pd.DataFrame:
    """Extract [s1..s6] raw score components before tanh/WHT."""
    if len(df) < MIN_TRAIN_BARS:
        return pd.DataFrame()
    close  = col(df, "Close")
    df_f   = build_filtered_df(df)
    cyc_s  = cycle_series(close)
    s1, s2, s3, s4 = adaptive_ichimoku_scores(df_f, cyc_s)
    s5 = vidya_score(df)
    s6 = volume_weighted_rsi(df)
    return pd.DataFrame({
        "s1":s1, "s2":s2, "s3":s3, "s4":s4, "s5":s5, "s6":s6,
    }, index=df.index).dropna()


# ── Student-t density ──────────────────────────────────────────────────────────

def student_t_log_pdf(
    X:   np.ndarray,   # (n, d)
    mu:  np.ndarray,   # (d,)
    sig: np.ndarray,   # (d, d)  — scale matrix (NOT covariance when ν<∞)
    nu:  float,
) -> np.ndarray:
    """
    Log-pdf of multivariate Student-t distribution.

    log p(x; μ, Σ, ν) = log Γ((ν+d)/2) - log Γ(ν/2)
                        - d/2 log(νπ) - 1/2 log|Σ|
                        - (ν+d)/2 log(1 + δ/ν)

    where δ = (x-μ)ᵀ Σ⁻¹ (x-μ)  [squared Mahalanobis distance]
    """
    n, d   = X.shape
    diff   = X - mu                          # (n, d)

    # Cholesky decomposition for stable inversion
    try:
        L      = np.linalg.cholesky(sig)
        log_det= 2 * np.sum(np.log(np.diag(L)))
        # Solve L @ z = diff.T  →  z = L⁻¹ diff.T
        z      = np.linalg.solve(L, diff.T)  # (d, n)
        delta  = np.sum(z**2, axis=0)        # (n,) — Mahalanobis² per point
    except np.linalg.LinAlgError:
        # Fallback: add regularisation
        sig_reg = sig + 1e-6 * np.eye(d)
        L       = np.linalg.cholesky(sig_reg)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        z       = np.linalg.solve(L, diff.T)
        delta   = np.sum(z**2, axis=0)

    log_pdf = (
        special.gammaln((nu + d) / 2)
        - special.gammaln(nu / 2)
        - (d / 2) * np.log(nu * np.pi)
        - 0.5 * log_det
        - ((nu + d) / 2) * np.log(1 + delta / nu)
    )
    return log_pdf   # (n,)


# ── EM algorithm for Student-t Mixture ────────────────────────────────────────

def fit_tmm(
    X:            np.ndarray,   # (n_bars, d) — standardised components
    n_components: int   = N_COMPONENTS,
    nu_init:      float = NU_INIT,
    max_iter:     int   = 100,
    tol:          float = 1e-4,
) -> dict:
    """
    Fit a mixture of Student-t distributions via EM.

    Returns:
        dict with keys: pi, mu, sigma, nu, responsibilities
        pi    : (K,)   — mixing proportions
        mu    : (K, d) — component means
        sigma : (K, d, d) — component scale matrices
        nu    : (K,)   — degrees of freedom per component
        resp  : (n, K) — soft responsibilities
    """
    n, d = X.shape
    K    = n_components

    # Initialise with K-Means++ style seeding
    rng   = np.random.default_rng(42)
    idx   = rng.choice(n, K, replace=False)
    mu    = X[idx].copy()                                   # (K, d)
    sigma = np.array([np.eye(d) for _ in range(K)])        # (K, d, d)
    pi    = np.ones(K) / K                                  # (K,)
    nu    = np.full(K, nu_init)                             # (K,)

    log_lik_prev = -np.inf

    for iteration in range(max_iter):
        # ── E-step ────────────────────────────────────────────────────────────

        # Log-responsibilities (log r_{tk})
        log_r = np.zeros((n, K))
        for k in range(K):
            log_r[:, k] = np.log(pi[k] + 1e-300) + student_t_log_pdf(
                X, mu[k], sigma[k], nu[k]
            )

        # Normalise (log-sum-exp for numerical stability)
        log_r_max = log_r.max(axis=1, keepdims=True)
        log_norm  = log_r_max + np.log(
            np.exp(log_r - log_r_max).sum(axis=1, keepdims=True) + 1e-300
        )
        log_lik   = log_norm.sum()
        r         = np.exp(log_r - log_norm)    # (n, K) — responsibilities

        # Robustness weights w_{tk} = (ν_k + d) / (ν_k + δ_{tk})
        W = np.zeros((n, K))
        for k in range(K):
            diff   = X - mu[k]
            try:
                L      = np.linalg.cholesky(sigma[k])
                z      = np.linalg.solve(L, diff.T)
                delta  = np.sum(z**2, axis=0)
            except np.linalg.LinAlgError:
                delta  = np.sum(diff**2, axis=1)
            W[:, k] = (nu[k] + d) / (nu[k] + delta)

        # Check convergence
        if abs(log_lik - log_lik_prev) < tol:
            log.debug(f"  EM converged at iteration {iteration+1}")
            break
        log_lik_prev = log_lik

        # ── M-step ────────────────────────────────────────────────────────────

        for k in range(K):
            r_k  = r[:, k]        # (n,)
            w_k  = W[:, k]        # (n,)
            rw_k = r_k * w_k      # (n,)  — combined weight

            # Mixing proportion
            n_k    = r_k.sum()
            pi[k]  = n_k / n

            # Mean update (responsibility × robustness weighted)
            denom  = rw_k.sum() + 1e-10
            mu[k]  = (rw_k[:, None] * X).sum(axis=0) / denom

            # Scale matrix update
            diff   = X - mu[k]                              # (n, d)
            sigma[k] = (rw_k[:, None, None]
                        * diff[:, :, None]
                        * diff[:, None, :]).sum(axis=0) / (n_k + 1e-10)
            # Regularise for numerical stability
            sigma[k] += 1e-5 * np.eye(d)

            # Degrees of freedom update via Newton's method on:
            # f(ν) = -ψ(ν/2) + log(ν/2) + 1 + mean_k[log(w) - w] + ψ((ν+d)/2) - log((ν+d)/2)
            # where ψ is the digamma function
            nu_k   = nu[k]
            for _ in range(10):   # Newton iterations
                e_w    = (r_k * W[:, k]).sum() / (n_k + 1e-10)
                e_logw = (r_k * np.log(W[:, k] + 1e-300)).sum() / (n_k + 1e-10)
                f  = (-special.digamma(nu_k/2)
                      + np.log(nu_k/2 + 1e-10) + 1
                      + e_logw - e_w
                      + special.digamma((nu_k + d)/2)
                      - np.log((nu_k + d)/2 + 1e-10))
                df = (-0.5*special.polygamma(1, nu_k/2)
                      + 1/(nu_k + 1e-10)
                      + 0.5*special.polygamma(1, (nu_k+d)/2)
                      - 1/(nu_k+d + 1e-10))
                nu_new = nu_k - f / (df + 1e-10)
                nu_k   = float(np.clip(nu_new, NU_MIN, NU_MAX))
            nu[k] = nu_k

    return {"pi": pi, "mu": mu, "sigma": sigma, "nu": nu, "resp": r}


# ── TMM training ───────────────────────────────────────────────────────────────

def train_tmm(
    data:         dict[str, pd.DataFrame],
    n_components: int  = N_COMPONENTS,
    forward_bars: int  = 5,
    save_path:    Path = MODEL_PATH,
) -> dict:
    """
    Train Student-t Mixture Model and estimate per-component score weights.

    Steps:
        1. Extract [s1..s6] components across training tickers
        2. Fit TMM via EM — Student-t robust to outlier score observations
        3. Compute soft responsibilities R[t,k] = P(component=k | x_t)
        4. Per-component weights = responsibility-weighted correlation
           of each score component with forward returns
        5. Softmax-sharpen weights to amplify meaningful differences
    """
    log.info(f"[TMM TRAIN]  Extracting components from {len(data)} tickers ...")

    from sklearn.preprocessing import StandardScaler

    all_comp, all_ret = [], []
    for ticker, df in data.items():
        try:
            comp = extract_components(df)
            if comp.empty or len(comp) < 100:
                continue
            close   = col(df, "Close").reindex(comp.index)
            fwd_ret = close.pct_change(forward_bars).shift(-forward_bars).reindex(comp.index)
            valid   = comp.join(fwd_ret.rename("fwd")).dropna()
            if len(valid) < 80:
                continue
            all_comp.append(valid[["s1","s2","s3","s4","s5","s6"]].values)
            all_ret.append(valid["fwd"].values)
        except Exception as e:
            log.debug(f"  {ticker}: {e}")

    if not all_comp:
        raise RuntimeError("No valid training data")

    X_raw = np.vstack(all_comp)
    R_ret = np.concatenate(all_ret)
    log.info(f"[TMM TRAIN]  {len(X_raw)} training bars")

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    # Fit Student-t mixture
    log.info(f"[TMM TRAIN]  Fitting Student-t mixture ({n_components} components) ...")
    result = fit_tmm(X, n_components=n_components)
    pi, mu, sigma, nu, R_soft = (
        result["pi"], result["mu"], result["sigma"],
        result["nu"], result["resp"],
    )

    log.info(f"[TMM TRAIN]  Mixing proportions: {pi.round(3)}")
    log.info(f"[TMM TRAIN]  Degrees of freedom ν: {nu.round(2)}")
    log.info(f"[TMM TRAIN]  (ν < 5: heavy tails, ν > 15: near-Gaussian)")

    # Estimate per-component weights
    NAMES = ["s1","s2","s3","s4","s5","s6"]
    weight_matrix = np.zeros((n_components, N_SCORE_COMP))

    for k in range(n_components):
        r_k   = R_soft[:, k]
        w_sum = r_k.sum() + 1e-10

        raw_corrs = np.zeros(N_SCORE_COMP)
        for j in range(N_SCORE_COMP):
            x_j  = X_raw[:, j]
            mu_x = (r_k * x_j).sum() / w_sum
            mu_r = (r_k * R_ret).sum() / w_sum
            cov  = (r_k * (x_j - mu_x) * (R_ret - mu_r)).sum() / w_sum
            vx   = (r_k * (x_j - mu_x)**2).sum() / w_sum
            vr   = (r_k * (R_ret - mu_r)**2).sum() / w_sum
            denom= np.sqrt(vx * vr)
            raw_corrs[j] = abs(cov / denom) if denom > 1e-10 else 0.0

        # Sharpen with temperature scaling before softmax
        # Higher temperature → more uniform weights
        # Lower temperature → winner-takes-more
        temperature = 0.05   # tuned so exp(corr/T) gives meaningful spread
        w = np.exp(raw_corrs / temperature)
        w = w / w.sum()
        weight_matrix[k] = w

        log.info(f"[TMM TRAIN]  Component {k} (ν={nu[k]:.1f}, π={pi[k]:.2f}) weights: "
                 f"{dict(zip(NAMES, w.round(3)))}")

    model = {
        "pi": pi, "mu": mu, "sigma": sigma, "nu": nu,
        "scaler":        scaler,
        "weight_matrix": weight_matrix,
        "n_components":  n_components,
        "forward_bars":  forward_bars,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"[TMM TRAIN]  Saved → {save_path}")
    return model


def load_tmm(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Inference ──────────────────────────────────────────────────────────────────

def compute_responsibilities(
    X:     np.ndarray,   # (n, d) standardised
    model: dict,
) -> np.ndarray:
    """Compute soft responsibilities R[t,k] for each bar."""
    pi, mu, sigma, nu = model["pi"], model["mu"], model["sigma"], model["nu"]
    K  = len(pi)
    n  = len(X)
    d  = X.shape[1]

    log_r = np.zeros((n, K))
    for k in range(K):
        log_r[:, k] = np.log(pi[k] + 1e-300) + student_t_log_pdf(
            X, mu[k], sigma[k], nu[k]
        )

    log_r_max = log_r.max(axis=1, keepdims=True)
    r = np.exp(log_r - log_r_max)
    r = r / (r.sum(axis=1, keepdims=True) + 1e-300)
    return r   # (n, K)


def tmm_composite_score(
    df:    pd.DataFrame,
    model: dict | None = None,
) -> pd.Series:
    """
    Student-t mixture weighted composite score.

    For each bar t:
        1. Get soft responsibilities r_t = P(component | x_t)  via Student-t density
        2. Blend component weight vectors: w(t) = Σ_k r_k(t) × W_k
           → smooth continuous interpolation, no hard regime switches
        3. raw(t) = N × dot(w(t), components(t))   [N=6 preserves scale]
        4. score(t) = tanh(raw(t)) × WHT_multiplier

    Robustness vs Gaussian: extreme score observations (outlier bars)
    get appropriate t-distribution likelihood rather than near-zero
    Gaussian likelihood, so responsibilities remain stable and meaningful.
    """
    if len(df) < MIN_TRAIN_BARS + WHT_WINDOW:
        return pd.Series(dtype=float)

    comp = extract_components(df)
    if comp.empty:
        return pd.Series(dtype=float)

    close = col(df, "Close")

    if model is None:
        # Fallback: equal weights = same as adaptive baseline
        raw  = comp.sum(axis=1)
        base = raw.apply(np.tanh)
        wht  = wht_multiplier_series(close).reindex(comp.index)
        return (base * wht).clip(-1, 1)

    scaler = model["scaler"]
    W      = model["weight_matrix"]   # (K, 6)

    X_raw  = comp.values
    X      = scaler.transform(X_raw)

    # Soft responsibilities per bar
    R_soft = compute_responsibilities(X, model)  # (n, K)

    # Blend weight vectors: w(t) = Σ_k r_k(t) × W_k  → (n, 6)
    blended_weights = R_soft @ W      # (n, K) @ (K, 6) = (n, 6)

    # Weighted sum per bar, scaled by N to preserve magnitude
    N          = N_SCORE_COMP
    raw_scores = N * (X_raw * blended_weights).sum(axis=1)

    raw   = pd.Series(raw_scores, index=comp.index)
    base  = raw.apply(np.tanh)
    wht   = wht_multiplier_series(close).reindex(comp.index)
    final = (base * wht).clip(-1, 1)

    return final


def tmm_latest_score(df: pd.DataFrame, model: dict | None = None) -> tuple[float, float]:
    """Return (score_now, score_prev) for live trading."""
    sc = tmm_composite_score(df, model).dropna()
    if len(sc) < 2:
        return 0.0, 0.0
    return float(sc.iloc[-1]), float(sc.iloc[-2])


# ── DBSCAN diagnostic ──────────────────────────────────────────────────────────

def run_dbscan_diagnostic(
    data:        dict[str, pd.DataFrame],
    eps:         float = 0.5,
    min_samples: int   = 30,
) -> dict:
    """
    Run DBSCAN on PCA-projected score components to characterise
    actual density structure — how many genuine clusters exist,
    and what fraction of data is noise.

    Interpretation guide:
        n_clusters=0, noise>80%  → purely continuous, no discrete structure
        n_clusters=1-2, noise~50% → weak clustering, GMM still better
        n_clusters=3+, noise<30% → genuine discrete regimes, HMM/GMM appropriate
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("scikit-learn required for DBSCAN diagnostic")
        return {}

    all_comp = []
    for df in data.values():
        comp = extract_components(df)
        if not comp.empty:
            all_comp.append(comp.values)

    if not all_comp:
        return {}

    X_raw = np.vstack(all_comp)
    X     = StandardScaler().fit_transform(X_raw)

    # Project to 2D for DBSCAN (curse of dimensionality in 6D)
    pca   = PCA(n_components=2)
    X_2d  = pca.fit_transform(X)
    var   = pca.explained_variance_ratio_.cumsum()[-1] * 100

    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(X_2d)
    labels = db.labels_
    n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise= (labels == -1).sum()
    noise_pct = n_noise / len(labels) * 100

    log.info(f"\n[DBSCAN]  PCA 2D captures {var:.1f}% of variance")
    log.info(f"[DBSCAN]  eps={eps}, min_samples={min_samples}")
    log.info(f"[DBSCAN]  Clusters found: {n_cl}")
    log.info(f"[DBSCAN]  Noise points:   {n_noise}/{len(labels)} ({noise_pct:.1f}%)")

    verdict = (
        "Data is CONTINUOUS → Student-t mixture is appropriate, "
        "hard clustering (K-Means/HMM) will overfit"
        if n_cl <= 2 or noise_pct > 50
        else f"{n_cl} genuine dense clusters found — discrete regime model may be viable"
    )
    log.info(f"[DBSCAN]  Verdict: {verdict}\n")
    return {"n_clusters": n_cl, "noise_pct": noise_pct, "verdict": verdict}