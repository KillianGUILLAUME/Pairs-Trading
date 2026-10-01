"""
research/backtest/simulation.py

Simulation Monte Carlo de paires cointégrées via processus OU + sauts de Lévy.
Pipeline : Simulate → Signal → Backtest → Métriques agrégées (VaR, CVaR, Sharpe).
Pas de modèle d'IA — uniquement des processus stochastiques classiques calibrés.

Usage:
    # Scénario unique (200 paths, régime normal, frais base)
    python -m research.backtest.simulation

    # Stress test complet (toutes combinaisons)
    python -m research.backtest.simulation --regime all --fees all --n_paths 500

    # Régime customisé
    python -m research.backtest.simulation --regime stressed --fees realistic --bars 3000
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import Parallel, delayed
from loguru import logger

# Assure l'import depuis la racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.signals.signal_generator import SignalGenerator
from research.backtest.engine import BacktestConfig, PerformanceMetrics, PositionManager, compute_kalman_halflife
from research.backtest.stress_test import FEE_SCENARIOS, RealisticCostModel, RealisticPnLEngine


# ============================================================
#  1. Configuration de la simulation
# ============================================================

@dataclass
class SimulationConfig:
    """Paramètres calibrables des processus stochastiques (tout en annualisé)."""
    
    # Volatilités annualisées
    sigma_a: float = 0.60
    sigma_b: float = 0.70
    
    # Drift (0 = martingale)
    mu_a: float = 0.0
    mu_b: float = 0.0
    
    # Spread OU
    theta: float = 0.05          # Vitesse de retour à la moyenne (par barre)
    mu_spread: float = 0.0       # Moyenne long terme
    sigma_spread: float = 0.01   # Volatilité du bruit OU
    
    # Sauts de Lévy (Poisson composé)
    jump_intensity: float = 0.01  # P(saut) par barre
    jump_mean: float = 0.0
    jump_std: float = 0.05
    
    # Student-t pour les fat tails du spread
    t_df: float = 5.0            # Degrés de liberté (∞ → Gaussien)
    
    # Corrélation des bruits
    rho: float = 0.85
    
    # Hedge ratio structurel
    beta: float = 1.0
    
    # Discrétisation
    dt: float = 1.0 / 8760      # 1 heure
    
    def __post_init__(self):
        self.vol_a_bar = self.sigma_a * np.sqrt(self.dt)
        self.vol_b_bar = self.sigma_b * np.sqrt(self.dt)


# ============================================================
#  2. Régimes de marché pré-calibrés
# ============================================================

MARKET_REGIMES = {
    "calm": SimulationConfig(
        sigma_a=0.40, sigma_b=0.45,
        theta=0.08, sigma_spread=0.005,
        jump_intensity=0.005, jump_std=0.02, t_df=10.0,
        rho=0.92,
    ),
    "normal": SimulationConfig(
        sigma_a=0.60, sigma_b=0.70,
        theta=0.05, sigma_spread=0.01,
        jump_intensity=0.01, jump_std=0.05, t_df=5.0,
        rho=0.85,
    ),
    "volatile": SimulationConfig(
        sigma_a=0.90, sigma_b=1.00,
        theta=0.03, sigma_spread=0.02,
        jump_intensity=0.02, jump_std=0.08, t_df=4.0,
        rho=0.75,
    ),
    "stressed": SimulationConfig(
        sigma_a=1.20, sigma_b=1.40,
        theta=0.01, sigma_spread=0.04,
        jump_intensity=0.05, jump_std=0.12, t_df=3.0,
        rho=0.60,
    ),
}


# ============================================================
#  2b. Bootstrap depuis les paramètres calibrés localement
# ============================================================

def load_calibrated_params(
    screened_parquet: str,
    max_adf_pvalue: float = 0.05,
) -> pd.DataFrame:
    """
    Charge les fenêtres "pépites" (ADF < seuil) depuis le Parquet screened
    et retourne les paramètres OU calibrés localement.
    
    Returns: DataFrame avec colonnes [ou_theta, ou_sigma, ou_mu, half_life, hurst, adf_pvalue]
    """
    df = pd.read_parquet(screened_parquet)
    gold = df[df["adf_pvalue"] < max_adf_pvalue].copy()
    
    if len(gold) == 0:
        raise ValueError(f"Aucune fenêtre stationnaire (ADF < {max_adf_pvalue}) dans {screened_parquet}")
    
    # Filtrer les calibrations aberrantes
    gold = gold[gold["ou_theta"] > 0]           # θ > 0 requis (mean-reversion)
    gold = gold[gold["ou_sigma"] > 1e-6]        # σ > 0
    gold = gold[gold["half_life"] < 1e6]         # half-life finie
    
    logger.info(f"📊 {len(gold)} fenêtres calibrées chargées depuis {screened_parquet}")
    logger.info(f"   θ ∈ [{gold['ou_theta'].min():.4f}, {gold['ou_theta'].max():.4f}]  "
                f"μ={gold['ou_theta'].mean():.4f} ± {gold['ou_theta'].std():.4f}")
    logger.info(f"   σ ∈ [{gold['ou_sigma'].min():.4f}, {gold['ou_sigma'].max():.4f}]  "
                f"μ={gold['ou_sigma'].mean():.4f} ± {gold['ou_sigma'].std():.4f}")
    
    return gold[["ou_theta", "ou_sigma", "ou_mu", "half_life", "hurst", "adf_pvalue"]]


def bootstrap_config(
    calibrated_params: pd.DataFrame,
    base_config: SimulationConfig = None,
    rng: np.random.Generator = None,
) -> SimulationConfig:
    """
    Pioche un couple (θ, σ) au hasard dans la distribution empirique
    et retourne un SimulationConfig avec ces paramètres injectés.
    """
    if base_config is None:
        base_config = MARKET_REGIMES["normal"]
    if rng is None:
        rng = np.random.default_rng()
    
    # Tirage aléatoire d'une fenêtre
    idx = rng.integers(len(calibrated_params))
    row = calibrated_params.iloc[idx]
    
    # Copie du config de base avec les paramètres locaux
    return SimulationConfig(
        sigma_a=base_config.sigma_a,
        sigma_b=base_config.sigma_b,
        mu_a=base_config.mu_a,
        mu_b=base_config.mu_b,
        theta=float(row["ou_theta"]),
        mu_spread=float(row["ou_mu"]),
        sigma_spread=float(row["ou_sigma"]),
        jump_intensity=base_config.jump_intensity,
        jump_mean=base_config.jump_mean,
        jump_std=base_config.jump_std,
        t_df=base_config.t_df,
        rho=base_config.rho,
        beta=base_config.beta,
    )


def run_monte_carlo_calibrated(
    screened_parquet: str,
    n_paths: int = 200,
    n_bars: int = 2000,
    p_a_init: float = 0.50,
    p_b_init: float = 35.0,
    base_regime: str = "normal",
    fee_scenario: str = "base",
    max_adf_pvalue: float = 0.05,
    bt_config: BacktestConfig = None,
    sig_config: dict = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo avec bootstrap des paramètres OU depuis les fenêtres calibrées.
    
    Pour chaque chemin simulé :
    1. Pioche un couple (θ, σ) dans la distribution empirique des fenêtres "pépites"
    2. Simule un chemin OU + Lévy avec ces paramètres locaux
    3. Applique les signaux et le backtest
    
    → Les chemins reflètent la VRAIE diversité des régimes observés.
    """
    calibrated = load_calibrated_params(screened_parquet, max_adf_pvalue)
    base_config = MARKET_REGIMES.get(base_regime, MARKET_REGIMES["normal"])
    
    if bt_config is None:
        bt_config = BacktestConfig(
            initial_capital=100_000, position_size=0.10,
            entry_threshold=2.0, exit_threshold=0.3,
        )
    
    logger.info(f"{'='*60}")
    logger.info(f"🎲 MONTE CARLO CALIBRÉ | {base_regime.upper()} × {fee_scenario}")
    logger.info(f"   {n_paths} chemins × {n_bars} barres")
    logger.info(f"   Bootstrap depuis {len(calibrated)} fenêtres pépites")
    logger.info(f"{'='*60}")
    
    # Génération avec bootstrap
    rng = np.random.default_rng(seed)
    t0 = time.time()
    
    def _sim_one(i):
        local_config = bootstrap_config(calibrated, base_config, np.random.default_rng(seed + i))
        return simulate_pair_ou_levy(n_bars, p_a_init, p_b_init, local_config, seed=seed + i + n_paths)
    
    paths = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_sim_one)(i) for i in range(n_paths)
    )
    logger.info(f"📈 {n_paths} chemins calibrés générés en {time.time()-t0:.1f}s")
    
    # Backtest
    t1 = time.time()
    sig_gen = SignalGenerator(
        zscore_window=(sig_config or {}).get("zscore_window", 168),
        entry_threshold=(sig_config or {}).get("entry_threshold", 2.0),
        exit_threshold=(sig_config or {}).get("exit_threshold", 0.3),
        compute_signature=False,
    )
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single_path)(
            i, paths[i][0], paths[i][1],
            bt_config, sig_gen, fee_scenario,
        )
        for i in range(n_paths)
    )
    logger.info(f"⚡ {n_paths} backtests terminés en {time.time()-t1:.1f}s")
    
    df = pd.DataFrame(results)
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err > 0:
            logger.warning(f"⚠️ {n_err}/{n_paths} chemins en erreur")
        df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")
    
    return df


# ============================================================
#  3. Générateur OU + Lévy
# ============================================================

def simulate_pair_ou_levy(
    n_bars: int,
    p_a_init: float,
    p_b_init: float,
    config: SimulationConfig,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simule une paire cointégrée :
    
    Actif A :  dlog(A) = (μ_a − ½σ²_a)dt + σ_a dW_a + J_a dN_a
    Spread  :  ds       = θ(μ − s)dt + σ_s dZ + J_s dN_s    (OU + sauts + Student-t)
    Actif B :  log(B)   = log(B_0) + β*(log(A) − log(A_0)) + s    (cointégration)
    
    Returns: (price_a, price_b) arrays de longueur n_bars
    """
    rng = np.random.default_rng(seed)
    c = config
    
    log_a = np.zeros(n_bars)
    log_b = np.zeros(n_bars)
    spread = np.zeros(n_bars)
    
    log_a_0 = np.log(p_a_init)
    log_b_0 = np.log(p_b_init)
    
    log_a[0] = log_a_0
    log_b[0] = log_b_0
    spread[0] = 0.0  # OU part de 0 (l'équilibre)
    
    # Cholesky pour la corrélation
    L = np.array([[1.0, 0.0], [c.rho, np.sqrt(max(1 - c.rho**2, 0))]])
    
    # Facteur de normalisation Student-t → variance unitaire
    scale_t = np.sqrt((c.t_df - 2) / c.t_df) if c.t_df > 2 else 1.0
    
    for t in range(1, n_bars):
        # Bruits corrélés
        z = rng.standard_normal(2)
        z_corr = L @ z
        
        # Sauts de Lévy sur l'actif A
        n_jumps = rng.poisson(c.jump_intensity)
        jump_a = rng.normal(c.jump_mean, c.jump_std) * n_jumps if n_jumps > 0 else 0.0
        
        # Spread OU avec bruit Student-t (fat tails) + sauts propres
        noise_spread = rng.standard_t(c.t_df) * scale_t
        jump_spread = rng.normal(0, c.jump_std * 0.5) * rng.poisson(c.jump_intensity * 0.3)
        
        ds = c.theta * (c.mu_spread - spread[t-1]) + c.sigma_spread * noise_spread + jump_spread
        spread[t] = spread[t-1] + ds
        
        # Actif A : GBM + sauts
        drift_a = (c.mu_a - 0.5 * c.sigma_a**2) * c.dt
        log_a[t] = log_a[t-1] + drift_a + c.vol_a_bar * z_corr[0] + jump_a
        
        # Actif B : forcé par cointegration
        # log(B_t) = log(B_0) + β * ( log(A_t) - log(A_0) ) + spread_t
        # → B suit A (beta-weighted) avec un écart OU autour de la relation
        noise_b = c.vol_b_bar * z_corr[1] * 0.10  # Bruit idiosyncratique résiduel
        log_b[t] = log_b_0 + c.beta * (log_a[t] - log_a_0) + spread[t] + noise_b
    
    return np.exp(log_a), np.exp(log_b)


# ============================================================
#  4. Pipeline unitaire : 1 chemin → métriques
# ============================================================

def _run_single_path(
    path_idx: int,
    price_a: np.ndarray,
    price_b: np.ndarray,
    bt_config: BacktestConfig,
    sig_gen: SignalGenerator,
    fee_scenario_name: str,
) -> dict:
    """Pipeline atomique : signal → positions → PnL → métriques."""
    
    timestamps = np.arange(len(price_a))
    
    try:
        # 1. Signaux Kalman + Z-score
        signal = sig_gen.generate(
            timestamps, price_a, price_b,
            symbol_a="SYNTH_A", symbol_b="SYNTH_B",
            auto_calibrate=True,
            use_regime_filter=False,
        )
        
        # 2. Positions avec half-life dynamique
        hl = compute_kalman_halflife(signal.spreads)
        dynamic_max_hold = max(int(hl * 2.0), 20)
        
        pm = PositionManager(bt_config)
        positions = pm.compute_positions(
            signal.entry_long, signal.entry_short,
            signal.exit_signal, dynamic_max_hold,
        )
        
        # 3. PnL avec coûts réalistes
        fee_scenario = FEE_SCENARIOS[fee_scenario_name]
        cost_model = RealisticCostModel(fee_scenario)
        pnl_engine = RealisticPnLEngine(bt_config, cost_model)
        
        df = pnl_engine.compute(
            positions, signal.spreads,
            signal.price_a, signal.price_b, signal.betas,
        )
        
        # 4. Sanitize NaN from Kalman warmup (first ~300 bars)
        df["net_pnl"] = df["net_pnl"].fillna(0.0)
        df["raw_pnl"] = df["raw_pnl"].fillna(0.0)
        df["cost_tx"] = df["cost_tx"].fillna(0.0)
        df["cost_fund"] = df["cost_fund"].fillna(0.0)
        df["capital"] = bt_config.initial_capital + df["net_pnl"].cumsum()
        
        # 5. Métriques
        metrics = PerformanceMetrics.compute(df)
        metrics["path_id"] = path_idx
        metrics["fee_scenario"] = fee_scenario_name
        metrics["half_life"] = round(hl, 1)
        
        total_fees = df["cost_tx"].sum()
        total_funding = df["cost_fund"].sum()
        metrics["total_fees_pct"] = round(total_fees / bt_config.initial_capital * 100, 3)
        metrics["total_funding_pct"] = round(total_funding / bt_config.initial_capital * 100, 3)
        metrics["cost_drag_pct"] = round((total_fees + total_funding) / bt_config.initial_capital * 100, 3)
        
        return metrics
        
    except Exception as e:
        return {
            "path_id": path_idx,
            "fee_scenario": fee_scenario_name,
            "sharpe": np.nan,
            "total_return_pct": np.nan,
            "max_dd": np.nan,
            "error": str(e),
        }


# ============================================================
#  5. Monte Carlo complet
# ============================================================

def run_monte_carlo(
    n_paths: int = 200,
    n_bars: int = 2000,
    p_a_init: float = 0.50,
    p_b_init: float = 35.0,
    regime: str = "normal",
    fee_scenario: str = "base",
    bt_config: BacktestConfig = None,
    sig_config: dict = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Pipeline Monte Carlo :
    1. Simule N paires cointégrées (OU + Lévy)
    2. Applique Kalman → Z-score → Signaux
    3. Backtest avec coûts réalistes
    4. Retourne les métriques par chemin
    
    Returns: DataFrame (1 row par chemin simulé)
    """
    if bt_config is None:
        bt_config = BacktestConfig(
            initial_capital=100_000, position_size=0.10,
            entry_threshold=2.0, exit_threshold=0.3,
        )
    
    sim_config = MARKET_REGIMES.get(regime, MARKET_REGIMES["normal"])
    
    logger.info(f"{'='*60}")
    logger.info(f"🎲 MONTE CARLO | {regime.upper()} × {fee_scenario}")
    logger.info(f"   {n_paths} chemins × {n_bars} barres | "
                f"σ_A={sim_config.sigma_a:.0%} σ_B={sim_config.sigma_b:.0%} "
                f"θ={sim_config.theta} λ_J={sim_config.jump_intensity}")
    logger.info(f"{'='*60}")
    
    # Génération des chemins
    t0 = time.time()
    paths = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(simulate_pair_ou_levy)(
            n_bars, p_a_init, p_b_init, sim_config, seed=seed + i
        )
        for i in range(n_paths)
    )
    logger.info(f"📈 {n_paths} chemins générés en {time.time()-t0:.1f}s")
    
    # Backtest parallèle
    t1 = time.time()
    sig_gen = SignalGenerator(
        zscore_window=(sig_config or {}).get("zscore_window", 168),
        entry_threshold=(sig_config or {}).get("entry_threshold", 2.0),
        exit_threshold=(sig_config or {}).get("exit_threshold", 0.3),
        compute_signature=False,
    )
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single_path)(
            i, paths[i][0], paths[i][1],
            bt_config, sig_gen, fee_scenario,
        )
        for i in range(n_paths)
    )
    logger.info(f"⚡ {n_paths} backtests terminés en {time.time()-t1:.1f}s")
    
    df = pd.DataFrame(results)
    
    # Filtrer les erreurs
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err > 0:
            logger.warning(f"⚠️ {n_err}/{n_paths} chemins en erreur")
        df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")
    
    return df


# ============================================================
#  6. Métriques de risque agrégées
# ============================================================

def compute_risk_metrics(df: pd.DataFrame) -> dict:
    """VaR, CVaR, probabilités sur la distribution Monte Carlo."""
    
    sharpes = df["sharpe"].dropna()
    returns = df["total_return_pct"].dropna()
    drawdowns = df["max_dd"].dropna()
    
    if len(sharpes) == 0 or len(returns) == 0:
        return {"n_paths": 0, "error": "No valid paths",
                "sharpe_mean": np.nan, "var_95_sharpe": np.nan,
                "return_mean_pct": np.nan, "cvar_95_return_pct": np.nan,
                "prob_loss_pct": np.nan}
    
    # VaR / CVaR à 95%
    var_95_sharpe = float(np.percentile(sharpes, 5))
    var_95_return = float(np.percentile(returns, 5))
    var_95_dd = float(np.percentile(drawdowns, 5))
    
    tail_returns = returns[returns <= var_95_return]
    tail_dd = drawdowns[drawdowns <= var_95_dd]
    
    cvar_95_return = float(tail_returns.mean()) if len(tail_returns) > 0 else np.nan
    cvar_95_dd = float(tail_dd.mean()) if len(tail_dd) > 0 else np.nan
    
    # Probabilités
    n = len(sharpes)
    
    # Métriques de coûts si disponibles
    cost_stats = {}
    if "cost_drag_pct" in df.columns:
        cost_stats["cost_drag_mean_pct"] = round(df["cost_drag_pct"].mean(), 3)
        cost_stats["cost_drag_max_pct"] = round(df["cost_drag_pct"].max(), 3)
    
    return {
        "n_paths": n,
        # Sharpe
        "sharpe_mean": round(float(sharpes.mean()), 3),
        "sharpe_median": round(float(sharpes.median()), 3),
        "sharpe_std": round(float(sharpes.std()), 3),
        "var_95_sharpe": round(var_95_sharpe, 3),
        # Rendement
        "return_mean_pct": round(float(returns.mean()), 2),
        "return_median_pct": round(float(returns.median()), 2),
        "var_95_return_pct": round(var_95_return, 2),
        "cvar_95_return_pct": round(cvar_95_return, 2),
        # Drawdown
        "max_dd_mean_pct": round(float(drawdowns.mean()), 2),
        "var_95_dd_pct": round(var_95_dd, 2),
        "cvar_95_dd_pct": round(cvar_95_dd, 2),
        # Probabilités
        "prob_loss_pct": round(float((returns < 0).mean()) * 100, 1),
        "prob_sharpe_neg_pct": round(float((sharpes < 0).mean()) * 100, 1),
        "prob_sharpe_gt1_pct": round(float((sharpes > 1.0).mean()) * 100, 1),
        # Trades
        "avg_trades": round(float(df["n_trades"].mean()), 0) if "n_trades" in df else None,
        "avg_pct_in_market": round(float(df["pct_in_market"].mean()), 1) if "pct_in_market" in df else None,
        # Coûts
        **cost_stats,
    }


def print_risk_report(risk: dict, regime: str, fee_scenario: str):
    """Rapport formaté."""
    print(f"\n{'='*65}")
    print(f"📊 RAPPORT MONTE CARLO — {regime.upper()} × {fee_scenario.upper()}")
    print(f"{'='*65}")
    print(f"   Chemins valides   : {risk['n_paths']}")
    print(f"")
    print(f"   ── SHARPE RATIO ──────────────────────────────")
    print(f"   Moyenne           : {risk['sharpe_mean']}")
    print(f"   Médiane           : {risk['sharpe_median']}")
    print(f"   Écart-type        : {risk['sharpe_std']}")
    print(f"   VaR 95%           : {risk['var_95_sharpe']}  (pire 5%)")
    print(f"   P(Sharpe > 1)     : {risk['prob_sharpe_gt1_pct']}%")
    print(f"   P(Sharpe < 0)     : {risk['prob_sharpe_neg_pct']}%")
    print(f"")
    print(f"   ── RENDEMENT ─────────────────────────────────")
    print(f"   Moyenne           : {risk['return_mean_pct']}%")
    print(f"   VaR 95%           : {risk['var_95_return_pct']}%")
    print(f"   CVaR 95%          : {risk['cvar_95_return_pct']}%")
    print(f"   P(perte)          : {risk['prob_loss_pct']}%")
    print(f"")
    print(f"   ── DRAWDOWN ──────────────────────────────────")
    print(f"   MDD moyen         : {risk['max_dd_mean_pct']}%")
    print(f"   VaR 95% MDD       : {risk['var_95_dd_pct']}%")
    print(f"   CVaR 95% MDD      : {risk['cvar_95_dd_pct']}%")
    if risk.get("avg_trades"):
        print(f"")
        print(f"   ── ACTIVITÉ ──────────────────────────────────")
        print(f"   Trades moyens     : {risk['avg_trades']:.0f}")
        print(f"   % temps en marché : {risk['avg_pct_in_market']}%")
    if risk.get("cost_drag_mean_pct"):
        print(f"   Coût moyen (drag) : {risk['cost_drag_mean_pct']}% du capital")
    print(f"{'='*65}")


# ============================================================
#  7. Stress Test multi-régimes
# ============================================================

def run_full_stress_test(
    n_paths: int = 200,
    n_bars: int = 2000,
    p_a_init: float = 0.50,
    p_b_init: float = 35.0,
    regimes: list = None,
    fee_scenarios: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Lance le Monte Carlo sur toutes les combinaisons (régime × frais).
    Retourne un DataFrame résumé indexé par (régime, frais).
    """
    if regimes is None:
        regimes = ["calm", "normal", "volatile", "stressed"]
    if fee_scenarios is None:
        fee_scenarios = ["optimistic", "base", "realistic"]
    
    all_risk = []
    
    for regime in regimes:
        for fee in fee_scenarios:
            df = run_monte_carlo(
                n_paths=n_paths, n_bars=n_bars,
                p_a_init=p_a_init, p_b_init=p_b_init,
                regime=regime, fee_scenario=fee, seed=seed,
            )
            risk = compute_risk_metrics(df)
            risk["regime"] = regime
            risk["fee_scenario"] = fee
            
            print_risk_report(risk, regime, fee)
            all_risk.append(risk)
    
    summary = pd.DataFrame(all_risk).set_index(["regime", "fee_scenario"])
    
    # Sauvegarde CSV
    out_dir = os.path.join(PROJECT_ROOT, "data", "stress_test_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"stress_test_{n_paths}p_{n_bars}b.csv")
    summary.to_csv(out_path)
    logger.info(f"\n💾 Résultats : {out_path}")
    
    return summary


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulation Monte Carlo OU + Lévy pour pairs trading"
    )
    parser.add_argument("--n_paths", type=int, default=200)
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--regime", type=str, default="normal",
                        choices=["calm", "normal", "volatile", "stressed", "all"])
    parser.add_argument("--fees", type=str, default="base",
                        choices=["optimistic", "base", "realistic", "stressed", "all"])
    parser.add_argument("--p_a", type=float, default=0.50)
    parser.add_argument("--p_b", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrated", action="store_true",
                        help="Bootstrap (θ,σ) depuis les fenêtres pépites du Parquet screened")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Chemin vers le fichier Parquet screened (avec --calibrated)")
    args = parser.parse_args()
    
    if args.calibrated:
        # Trouver le Parquet automatiquement si pas spécifié
        if args.parquet is None:
            args.parquet = os.path.join(
                PROJECT_ROOT, "data", "storage", "screened",
                "screened_ADA_USDTxAVAX_USDT_1h_w128_s8.parquet"
            )
        
        df = run_monte_carlo_calibrated(
            screened_parquet=args.parquet,
            n_paths=args.n_paths, n_bars=args.bars,
            p_a_init=args.p_a, p_b_init=args.p_b,
            base_regime=args.regime, fee_scenario=args.fees, seed=args.seed,
        )
        risk = compute_risk_metrics(df)
        print_risk_report(risk, f"calibrated({args.regime})", args.fees)
    
    elif args.regime == "all" or args.fees == "all":
        regimes = list(MARKET_REGIMES.keys()) if args.regime == "all" else [args.regime]
        fees = ["optimistic", "base", "realistic"] if args.fees == "all" else [args.fees]
        
        summary = run_full_stress_test(
            n_paths=args.n_paths, n_bars=args.bars,
            p_a_init=args.p_a, p_b_init=args.p_b,
            regimes=regimes, fee_scenarios=fees, seed=args.seed,
        )
        
        print(f"\n{'='*65}")
        print("📋 TABLEAU RÉCAPITULATIF")
        print(f"{'='*65}")
        cols = ["sharpe_mean", "var_95_sharpe", "return_mean_pct", "cvar_95_return_pct", "prob_loss_pct"]
        print(summary[[c for c in cols if c in summary.columns]].to_string())
    else:
        df = run_monte_carlo(
            n_paths=args.n_paths, n_bars=args.bars,
            p_a_init=args.p_a, p_b_init=args.p_b,
            regime=args.regime, fee_scenario=args.fees, seed=args.seed,
        )
        risk = compute_risk_metrics(df)
        print_risk_report(risk, args.regime, args.fees)