import numpy as np
import pandas as pd

def generate_synthetic_pair(
    n_bars: int = 5000,
    p_a_init: float = 100.0,
    p_b_init: float = 100.0,
    vol_a: float = 0.00888,
    spread_std: float = 0.04889,
    spread_hl: float = 20.0,
    beta_mean: float = 1.04,
    beta_std: float = 0.0956
) -> pd.DataFrame:
    """
    Génère une paire de prix artificiellement co-intégrée.
    """
    # 1. Génération de l'Actif A (Geometric Brownian Motion)
    # On simule les rendements log-normaux puis on en fait la somme cumulée
    mu_a = 0.0 
    returns_a = np.random.normal(mu_a - (vol_a**2)/2, vol_a, n_bars)
    log_p_a = np.log(p_a_init) + np.cumsum(returns_a)
    price_a = np.exp(log_p_a)
    
    # 2. Génération du Spread (Processus d'Ornstein-Uhlenbeck)
    # Méthode d'Euler-Maruyama pour la simulation discrète
    theta_beta = 0.002
    sigma_beta = beta_std * np.sqrt(2 * theta_beta)
    
    beta = np.zeros(n_bars)
    beta[0] = beta_mean
    noise_beta = np.random.normal(0, 1, n_bars)
    for t in range(1, n_bars):
        beta[t] = beta[t-1] + theta_beta * (beta_mean - beta[t-1]) + sigma_beta * noise_beta[t]

    theta_spread = np.log(2) / spread_hl
    sigma_spread = spread_std * np.sqrt(2 * theta_spread)
    
    spread = np.zeros(n_bars)
    noise_spread = np.random.normal(0, 1, n_bars)
    for t in range(1, n_bars):
        spread[t] = spread[t-1] - theta_spread * spread[t-1] + sigma_spread * noise_spread[t]
        
    # 3. Génération de l'Actif B (Co-intégré à A)
    # log(Pb) = beta * log(Pa) + spread
    log_p_b = np.log(p_b_init) + beta * (log_p_a - np.log(p_a_init)) + spread
    price_b = np.exp(log_p_b)
    
    timestamps = np.arange(n_bars)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "SYNTH_A": price_a,
        "SYNTH_B": price_b,
        "true_beta": beta,
        "true_spread": spread
    }).set_index("timestamp")
    
    return df


def generate_crypto_synthetic_pair(
    n_bars: int = 5000,
    p_a_init: float = 100.0,
    p_b_init: float = 100.0,
    vol_a: float = 0.00888,
    spread_std: float = 0.04889,
    spread_hl: float = 20.0,
    beta_mean: float = 1.04,
    beta_std: float = 0.0956,
    jump_prob: float = 0.005,      # Probabilité de 0.5% d'avoir un flash crash/pump par barre
    jump_vol: float = 0.04,        # Volatilité du saut (ex: bougie de 4% d'un coup)
    t_df: float = 3.0              # Degrés de liberté Student-t (plus c'est bas, plus c'est extrême)
) -> pd.DataFrame:
    """
    Génère une paire crypto avec Jumps (Poisson) et queues épaisses (Lévy/Student).
    Utile pour simuler des marchés à forte volatilité et avec des événements extrêmes.
    """
    
    # ==========================================
    # 1. ACTIF A : Jump-Diffusion (Modèle de Merton)
    # ==========================================
    # Bruit normal (Brownien)
    normal_returns = np.random.normal(0 - (vol_a**2)/2, vol_a, n_bars)
    
    # Processus de sauts (Poisson + Gaussienne pour la taille du saut)
    poisson_jumps = np.random.poisson(jump_prob, n_bars)
    jump_sizes = np.random.normal(0, jump_vol, n_bars) * poisson_jumps
    
    # Rendements totaux = Diffusion + Sauts
    returns_a = normal_returns + jump_sizes
    log_p_a = np.log(p_a_init) + np.cumsum(returns_a)
    price_a = np.exp(log_p_a)
    
    # ==========================================
    # 2. BETA : Processus de dérive lente
    # ==========================================
    theta_beta = 0.002
    sigma_beta = beta_std * np.sqrt(2 * theta_beta)
    
    beta = np.zeros(n_bars)
    beta[0] = beta_mean
    noise_beta = np.random.normal(0, 1, n_bars)
    for t in range(1, n_bars):
        beta[t] = beta[t-1] + theta_beta * (beta_mean - beta[t-1]) + sigma_beta * noise_beta[t]
        
    # ==========================================
    # 3. SPREAD : Ornstein-Uhlenbeck "Fat-Tailed"
    # ==========================================
    theta_spread = np.log(2) / spread_hl
    sigma_spread = spread_std * np.sqrt(2 * theta_spread)
    
    spread = np.zeros(n_bars)
    
    # On génère du bruit de Student-t (Fat tails)
    # On normalise la variance à 1 pour que sigma_spread contrôle l'échelle exacte
    scale_factor = np.sqrt((t_df - 2) / t_df) if t_df > 2 else 1.0
    noise_spread_t = np.random.standard_t(t_df, n_bars) * scale_factor
    
    for t in range(1, n_bars):
        spread[t] = spread[t-1] - theta_spread * spread[t-1] + sigma_spread * noise_spread_t[t]
        
    # ==========================================
    # 4. ACTIF B : Co-intégration
    # ==========================================
    log_p_b = np.log(p_b_init) + beta * (log_p_a - np.log(p_a_init)) + spread
    price_b = np.exp(log_p_b)
    
    timestamps = np.arange(n_bars)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "SYNTH_A": price_a,
        "SYNTH_B": price_b,
        "true_beta": beta,
        "true_spread": spread
    }).set_index("timestamp")
    
    return df