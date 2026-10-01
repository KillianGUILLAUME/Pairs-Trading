import numpy as np
import torch
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

def compute_hurst_exponent(series: np.ndarray) -> float:
    n = len(series)
    if n < 20: return 0.5
    
    max_k = min(n // 2, 128)
    sizes = []
    rs_values = []
    
    for size in [16, 20, 24, 32, 40, 48, 64, 80, 96, max_k]:
        if size > max_k or size < 8: continue
        n_chunks = n // size
        if n_chunks < 1: continue
            
        rs_list = []
        for i in range(n_chunks):
            chunk = series[i * size : (i + 1) * size]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 1e-10: rs_list.append(R / S)
        
        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))
    
    if len(sizes) < 3: return 0.5
    
    log_sizes, log_rs = np.log(sizes), np.log(rs_values)
    try:
        coeffs = np.polyfit(log_sizes, log_rs, 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))
    except Exception:
        return 0.5

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

def compute_tail_risk(returns: np.ndarray) -> float:
    if len(returns) < 10: return 30.0
    try:
        nu, loc, scale = student_t.fit(returns)
        return float(np.clip(nu, 1.0, 30.0))
    except Exception:
        return 30.0

def compute_asymmetry(returns: np.ndarray) -> float:
    if len(returns) < 10: return 0.0
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

def get_screener_feature_dict(log_a: np.ndarray, log_b: np.ndarray, log_btc: np.ndarray, log_eth: np.ndarray) -> dict:
    """ Computes and returns the dictionary of all generative features for the dataset """
    spread = log_a - log_b
    returns = np.diff(spread)
    
    adf_pvalue = compute_adf_pvalue(spread)
    hurst = compute_hurst_exponent(spread)
    half_life = compute_half_life(spread)
    ou_params = compute_ou_mle(spread, dt=1.0)
    nu = compute_tail_risk(returns)
    alpha = compute_asymmetry(returns)
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

def get_inference_tensor(
    log_a: np.ndarray, 
    log_b: np.ndarray, 
    log_btc: np.ndarray, 
    log_eth: np.ndarray,
    scaler_mean: np.ndarray = None,
    scaler_std: np.ndarray = None
) -> torch.Tensor:
    """
    Computes all features on the fly and returns the 1x10 conditional tensor.
    Sert au conditionnement "Live" du modèle SDE (Génération/Walk-Forward).
    """
    d = get_screener_feature_dict(log_a, log_b, log_btc, log_eth)

    if scaler_mean is None or scaler_std is None:
        raw_points = np.stack([log_a, log_b], axis=-1)
        scaler_mean = np.mean(raw_points, axis=0)
        scaler_std = np.std(raw_points, axis=0)
        scaler_std = np.where(scaler_std == 0, 1e-8, scaler_std)
        
    # On isole les prix au temps T=0
    price_a_0 = log_a[0]
    price_b_0 = log_b[0]
    
    # On applique la standardisation exacte de l'entraînement
    norm_a_0 = (price_a_0 - scaler_mean[0]) / scaler_std[0]
    norm_b_0 = (price_b_0 - scaler_mean[1]) / scaler_std[1]
    
    # On calcule le spread normalisé
    initial_spread = norm_a_0 - norm_b_0

    d["initial_spread"] = initial_spread
    
    metrics = torch.tensor([
        d["adf_pvalue"],
        d["hurst"],
        min(d["half_life"], 500.0),
        d["ou_theta"],
        d["ou_mu"],
        d["ou_sigma"],
        d["btc_return_24h"],
        d["btc_vol_72h"],
        d["eth_btc_ratio"],
        d["student_nu"],
        d["skew_alpha"],
        d["jump_lambda"],
        d["jump_sigma"],
        d["initial_spread"]
    ], dtype=torch.float32).unsqueeze(0)  # Shape (1, 13)
    
    return metrics