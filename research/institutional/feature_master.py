import numpy as np
import iisignature
from typing import Dict, List
from scipy.stats import pearsonr
from features.features_eng import (
    compute_adf_pvalue, 
    compute_hurst_exponent, 
    compute_half_life, 
    compute_ou_mle,
    compute_tail_risk,
    compute_asymmetry,
    compute_merton_jumps_pseudo_mle
)

class FeatureMaster:
    """
    Générateur de Features "Grade Institutionnel V3".
    Fusionne :
    - Path Signatures (IISignature)
    - Statistiques Classiques (ADF, Hurst, Half-Life)
    - Risques de Queue (Student-T nu, Skew, Jumps)
    - Microstructure (Lead-Lag)
    """
    def __init__(self, signature_level: int = 2):
        self.sig_level = signature_level

    def compute_signature(self, spread_path: np.ndarray) -> np.ndarray:
        n = len(spread_path)
        times = np.linspace(0, 1, n)
        path = np.column_stack([times, spread_path])
        return iisignature.sig(path, self.sig_level)

    def compute_lead_lag(self, ret_a: np.ndarray, ret_b: np.ndarray, max_lag: int = 5) -> Dict[str, float]:
        lags = range(-max_lag, max_lag + 1)
        corrs = {}
        for lag in lags:
            try:
                if lag < 0: c, _ = pearsonr(ret_a[-lag:], ret_b[:lag])
                elif lag > 0: c, _ = pearsonr(ret_a[:-lag], ret_b[lag:])
                else: c, _ = pearsonr(ret_a, ret_b)
                corrs[f"lag_{lag}"] = c if np.isfinite(c) else 0.0
            except:
                corrs[f"lag_{lag}"] = 0.0
        asymmetry = corrs.get("lag_1", 0) - corrs.get("lag_-1", 0)
        return {"lead_lag_asymmetry": asymmetry}

    def get_feature_vector(self, log_a: np.ndarray, log_b: np.ndarray) -> Dict[str, float]:
        spread = log_a - log_b
        ret_a = np.diff(log_a)
        ret_b = np.diff(log_b)
        ret_spread = np.diff(spread)
        
        # 1. Path Features
        sig = self.compute_signature(spread)
        ll = self.compute_lead_lag(ret_a, ret_b)
        
        # 2. Statistical Features (Legacy)
        adf = compute_adf_pvalue(spread)
        hurst = compute_hurst_exponent(spread)
        hl = compute_half_life(spread)
        ou = compute_ou_mle(spread)
        
        # 3. Risk Features
        nu = compute_tail_risk(ret_spread)
        alpha = compute_asymmetry(ret_spread)
        l_jump, s_jump = compute_merton_jumps_pseudo_mle(ret_spread)
        
        features = {
            "hurst": hurst,
            "adf_pvalue": adf,
            "half_life": min(hl, 500.0),
            "ou_theta": ou["theta"],
            "ou_mu": ou["mu"],
            "ou_sigma": ou["sigma"],
            "student_nu": nu,
            "skew_alpha": alpha,
            "jump_lambda": l_jump,
            "lead_lag_asymmetry": ll["lead_lag_asymmetry"]
        }
        
        for i, s_val in enumerate(sig):
            features[f"sig_{i}"] = s_val
            
        return features
