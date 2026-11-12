import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import t as student_t
from scipy.stats import skewnorm

# ============================================================
# Métriques Quantitatives et Statistiques
# ============================================================

def compute_adf_pvalue(spread: np.ndarray) -> float:
    try:
        result = adfuller(spread, maxlag=1, regression="c", autolag=None)
        return float(result[1])
    except Exception:
        return 1.0 

def _hurst_rs(increments: np.ndarray) -> float:
    """Analyse R/S (Hurst 1951) appliquée aux INCRÉMENTS d'un processus."""
    n = len(increments)
    if n < 20:
        return 0.5
    max_k = min(n // 2, 128)
    sizes, rs_values = [], []
    for size in [16, 20, 24, 32, 40, 48, 64, 80, 96, max_k]:
        if size > max_k or size < 8:
            continue
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = increments[i * size:(i + 1) * size]
            deviations = np.cumsum(chunk - np.mean(chunk))
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))
    if len(sizes) < 3:
        return 0.5
    try:
        coeffs = np.polyfit(np.log(sizes), np.log(rs_values), 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))
    except Exception:
        return 0.5


def _hurst_variogram(series: np.ndarray, max_lag: int = 32) -> float:
    """
    Exposant de Hurst généralisé par variogramme (sur les NIVEAUX) :
        τ(ℓ) = sqrt(Var(S_{t+ℓ} - S_t)),   log τ(ℓ) ≈ H log ℓ + C.
    Marche aléatoire -> 0.5 ; OU (variogramme saturant) -> < 0.5 ; tendance -> > 0.5.
    C'est l'estimateur décrit dans le README (§2.5).
    """
    n = len(series)
    if n < 20:
        return 0.5
    lags = [lag for lag in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if lag <= min(max_lag, n // 4)]
    if len(lags) < 3:
        return 0.5
    taus = []
    for lag in lags:
        d = series[lag:] - series[:-lag]
        # Moment d'ordre 2 NON centré (variogramme classique) : sans biais pour une
        # marche aléatoire ; np.var (centré) sous-estime aux grands lags à cause du
        # chevauchement des différences.
        v = np.mean(d ** 2)
        taus.append(np.sqrt(v) if v > 1e-18 else np.nan)
    taus = np.array(taus)
    ok = np.isfinite(taus)
    if ok.sum() < 3:
        return 0.5
    try:
        slope = np.polyfit(np.log(np.array(lags)[ok]), np.log(taus[ok]), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def compute_hurst_exponent(series: np.ndarray, method: str = "variogram") -> float:
    """
    Exposant de Hurst d'un processus donné par ses NIVEAUX (ex. le spread).
        H < 0.5 : retour à la moyenne ; H = 0.5 : marche aléatoire ; H > 0.5 : tendance.

    method="variogram" (défaut) : variogramme sur les niveaux (README §2.5).
    method="rs"                 : analyse R/S appliquée aux incréments np.diff(series).

    Correctif (audit 2026-11) : l'ancienne version appliquait R/S directement aux
    niveaux, ce qui traite le spread comme une série d'incréments et renvoie
    H ≈ 0.9-1.0 même pour un OU fortement mean-reverting.
    """
    series = np.asarray(series, dtype=float)
    if method == "rs":
        return _hurst_rs(np.diff(series))
    return _hurst_variogram(series)


def compute_half_life(spread: np.ndarray) -> float:
    n = len(spread)
    if n < 10: return float("inf")
    
    y = np.diff(spread)
    x = spread[:-1]
    
    x_mean, y_mean = np.mean(x), np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    
    if ss_xx < 1e-12: return float("inf")
    phi = ss_xy / ss_xx
    if phi >= 0: return float("inf")
    
    try:
        hl = -np.log(2) / np.log(1 + phi)
        return float(max(hl, 0.5))
    except (ValueError, ZeroDivisionError):
        return float("inf")

def compute_ou_mle(spread: np.ndarray, dt: float = 1.0) -> dict:
    n = len(spread)
    if n < 10: return {"theta": 0.0, "mu": 0.0, "sigma": 0.0}
    
    y = spread[1:]
    x = spread[:-1]
    
    n_obs = len(y)
    sx, sy = np.sum(x), np.sum(y)
    sxx, sxy = np.sum(x**2), np.sum(x * y)
    
    denom = n_obs * sxx - sx**2
    if abs(denom) < 1e-15:
        return {"theta": 0.0, "mu": float(np.mean(spread)), "sigma": float(np.std(np.diff(spread)))}
    
    b = (n_obs * sxy - sx * sy) / denom
    a = (sy - b * sx) / n_obs
    
    residuals = y - a - b * x
    sigma_eps = np.sqrt(np.mean(residuals**2))
    
    if b <= 0 or b >= 1:
        if b <= 0:
            b_safe = max(b, 1e-6)
            theta = min(-np.log(max(b_safe, 1e-10)) / dt, 10.0)
        else:
            theta = 0.0
        mu, sigma = float(np.mean(spread)), float(np.std(np.diff(spread)))
    else:
        theta = -np.log(b) / dt
        mu = a / (1 - b)
        exp_term = 1.0 - np.exp(-2 * theta * dt)
        if exp_term > 1e-10:
            sigma = sigma_eps * np.sqrt(2 * theta / exp_term)
        else:
            sigma = sigma_eps / np.sqrt(dt)
            
    return {"theta": round(float(theta), 6), "mu": round(float(mu), 6), "sigma": round(float(sigma), 6)}

def compute_tail_risk(returns: np.ndarray, fast: bool = False) -> float:
    """
    Degrés de liberté ν d'une Student-t ajustée aux rendements (queues épaisses si ν petit).
    fast=False : MLE scipy (~50 ms) ; fast=True : méthode des moments, ν = 4 + 6/κ_excès
    (ν clippé à [2.1, 30]).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 10: return 30.0
    if fast:
        x = returns - np.mean(returns)
        m2 = np.mean(x ** 2)
        if m2 < 1e-18: return 30.0
        kurt_excess = np.mean(x ** 4) / (m2 ** 2) - 3.0
        if kurt_excess <= 0.2: return 30.0
        return float(np.clip(4.0 + 6.0 / kurt_excess, 2.1, 30.0))
    try:
        nu, loc, scale = student_t.fit(returns)
        return float(np.clip(nu, 1.0, 30.0))
    except Exception:
        return 30.0

def compute_asymmetry(returns: np.ndarray, fast: bool = False) -> float:
    """
    Asymétrie des rendements. fast=False : paramètre α d'une skew-normale (MLE scipy, ~50 ms) ;
    fast=True : coefficient d'asymétrie empirique (moment d'ordre 3 standardisé), clippé à [-10, 10].
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 10: return 0.0
    if fast:
        x = returns - np.mean(returns)
        s = np.sqrt(np.mean(x ** 2))
        if s < 1e-12: return 0.0
        return float(np.clip(np.mean(x ** 3) / s ** 3, -10.0, 10.0))
    try:
        a, loc, scale = skewnorm.fit(returns)
        return float(np.clip(a, -10.0, 10.0))
    except Exception:
        return 0.0

def compute_merton_jumps_pseudo_mle(returns: np.ndarray) -> tuple:
    n = len(returns)
    if n < 10: return 0.0, 0.0
    
    median = np.median(returns)
    mad = np.median(np.abs(returns - median))
    sigma_cont_est = 1.4826 * mad
    if sigma_cont_est < 1e-8: sigma_cont_est = np.std(returns) + 1e-8
        
    threshold = 3.0 * sigma_cont_est
    jump_mask = np.abs(returns) > threshold
    jumps = returns[jump_mask]
    
    lambda_jump = len(jumps) / n
    if len(jumps) > 1: sigma_jump = float(np.std(jumps))
    elif len(jumps) == 1: sigma_jump = float(np.abs(jumps[0]))
    else: sigma_jump = 0.0
        
    return float(lambda_jump), float(sigma_jump)

# ============================================================
# API de Haut-Niveau (Wrappers)
# ============================================================

SCREENER_FEATURE_KEYS = [
    "adf_pvalue", "hurst", "half_life", "ou_theta", "ou_mu", "ou_sigma",
    "btc_return_24h", "btc_vol_72h", "eth_btc_ratio",
    "student_nu", "skew_alpha", "jump_lambda", "jump_sigma",
]

def get_window_diagnostics(log_a: np.ndarray, log_b: np.ndarray) -> dict:
    """
    Diagnostics de stationnarité d'une fenêtre (ADF, Hurst, demi-vie, OU).
    Utilisé par le screener sur la fenêtre CIBLE, pour la curation du dataset
    uniquement : ces valeurs ne doivent jamais servir de conditionnement.
    """
    spread = log_a - log_b
    ou = compute_ou_mle(spread, dt=1.0)
    return {
        "adf_pvalue": compute_adf_pvalue(spread),
        "hurst": compute_hurst_exponent(spread),
        "half_life": compute_half_life(spread),
        "ou_theta": ou["theta"], "ou_mu": ou["mu"], "ou_sigma": ou["sigma"],
    }

def get_screener_feature_dict(log_a: np.ndarray, log_b: np.ndarray, log_btc: np.ndarray, log_eth: np.ndarray,
                              fast: bool = False) -> dict:
    """
    Descripteurs d'une fenêtre de log-prix (13 clés, ordre `SCREENER_FEATURE_KEYS`).
    Les features macro (BTC / ETH) sont évaluées à la FIN de la fenêtre : pour un
    conditionnement causal, passer la fenêtre PASSÉE [t-W, t).
    fast=True : estimateurs de queue/asymétrie par moments (screener, ~10x plus rapide).
    """
    spread = log_a - log_b
    returns = np.diff(spread)
    
    adf_pvalue = compute_adf_pvalue(spread)
    hurst = compute_hurst_exponent(spread)
    half_life = compute_half_life(spread)
    ou_params = compute_ou_mle(spread, dt=1.0)
    nu = compute_tail_risk(returns, fast=fast)
    alpha = compute_asymmetry(returns, fast=fast)
    lambda_jump, sigma_jump = compute_merton_jumps_pseudo_mle(returns)
    
    n_bars = len(log_btc)
    idx_24 = max(0, n_bars - 25)
    idx_72 = max(0, n_bars - 73)
    
    btc_return_24h = float(log_btc[-1] - log_btc[idx_24]) if n_bars > 24 else 0.0
    btc_vol_72h = float(np.std(np.diff(log_btc[idx_72:]))) if n_bars > 72 else 0.0
    eth_btc_ratio = float(log_eth[-1] - log_btc[-1]) if len(log_eth) > 0 and len(log_btc) > 0 else 0.0
    
    return {
        "adf_pvalue": adf_pvalue,
        "hurst": hurst,
        "half_life": half_life,
        "ou_theta": ou_params["theta"],
        "ou_mu": ou_params["mu"],
        "ou_sigma": ou_params["sigma"],
        "btc_return_24h": btc_return_24h,
        "btc_vol_72h": btc_vol_72h,
        "eth_btc_ratio": eth_btc_ratio,
        "student_nu": nu,
        "skew_alpha": alpha,
        "jump_lambda": lambda_jump,
        "jump_sigma": sigma_jump,
    }

def get_inference_tensor(*args, **kwargs):
    """
    OBSOLÈTE (audit 2026-11) : renvoyait des features brutes non standardisées avec
    un scaling par asset, incompatible avec l'entraînement (z-score des conditions +
    ancrage global). Utiliser `GenerativeSDEEngine.conditions_from_history(...)`,
    qui applique exactement la standardisation stockée dans le checkpoint.
    """
    raise NotImplementedError(
        "get_inference_tensor est obsolète : utiliser "
        "research.backtest.generative_sde_engine.GenerativeSDEEngine.conditions_from_history"
    )
