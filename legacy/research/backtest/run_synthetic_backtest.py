import numpy as np
import pandas as pd
from loguru import logger
from research.backtest.generative_sde_engine import GenerativeSDEEngine
from research.backtest.engine import BacktestEngine, BacktestConfig
from research.signals.signal_generator import SignalGenerator, SignalGeneratorConfig

def run_monte_carlo_backtest():
    # 1. Configuration
    # Mets le chemin du modèle que tu viens de sauvegarder !
    MODEL_PATH = "../data/models/neural_sde_20260330_145500_ADA_USDTxAVAX_USDT.pt" 
    N_SCENARIOS = 200
    BARS = 2000 # Environ 3 mois de data en 1h
    
    # Prix initiaux au pif (ex: ADA = 0.5$, AVAX = 35$)
    P_A_INIT = 0.5
    P_B_INIT = 35.0

    # 2. Initialisation des Moteurs
    sde_engine = GenerativeSDEEngine(model_path=MODEL_PATH)
    bt_engine  = BacktestEngine(BacktestConfig(initial_capital=100_000, position_size=0.10))
    sig_gen = SignalGenerator(zscore_window=12, entry_threshold=2.5, exit_threshold=0.75)

    # 3. Génération de TOUS les univers parallèles (Très rapide sur GPU)
    synthetic_markets = sde_engine.simulate_markets(
        n_paths=N_SCENARIOS, 
        bars=BARS, 
        p_a_init=P_A_INIT, 
        p_b_init=P_B_INIT
    )

    all_metrics = []

    # 4. Backtest sur chaque univers généré
    for idx, market_df in enumerate(synthetic_markets):
        timestamps = market_df.index.values
        price_a = market_df["SYNTH_A"].values
        price_b = market_df["SYNTH_B"].values
        
        # Génération des signaux Z-Score via ton algorithme classique
        signal = sig_gen.generate(
            timestamps, price_a, price_b, 
            symbol_a="SYNTH_A", symbol_b="SYNTH_B"
        )
        
        # Execution du Backtest (ton fichier engine.py)
        res = bt_engine.run(signal, label=f"Scenario_{idx+1}")
        
        metrics = res["metrics"]
        metrics["Scenario"] = idx + 1
        all_metrics.append(metrics)

    # ================================================
    # 5. AGRÉGATION ET ANALYSE STATISTIQUE MONTE CARLO
    # ================================================
    results_df = pd.DataFrame(all_metrics).set_index("Scenario")
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Calcul des probabilités et VaR (Value at Risk) ---
    # Probabilité de perdre de l'argent (Return < 0)
    prob_loss = (results_df['total_return_pct'] < 0).mean() * 100
    
    # VaR 95% : Dans 95% des cas, la métrique sera PIRE que cette valeur
    var_95_sharpe = np.percentile(results_df['sharpe'], 5)
    var_95_return = np.percentile(results_df['total_return_pct'], 5)
    var_95_dd     = np.percentile(results_df['max_dd'], 5) # Le pire DD à 95% de confiance
    
    # Expected Shortfall (CVaR) : La moyenne des 5% des pires cas
    cvar_95_return = results_df[results_df['total_return_pct'] <= var_95_return]['total_return_pct'].mean()
    cvar_95_dd = results_df[results_df['max_dd'] <= var_95_dd]['max_dd'].mean()

    logger.info("\n" + "="*50)
    logger.info(f"📊 ANALYSE MONTE CARLO ({N_SCENARIOS} UNIVERS PARALLÈLES)")
    logger.info("="*50)
    logger.info(f"Moyenne (Expected) Sharpe : {results_df['sharpe'].mean():.3f}")
    logger.info(f"Médiane Sharpe            : {results_df['sharpe'].median():.3f}")
    logger.info(f"Probabilité de perte      : {prob_loss:.1f}%")
    logger.info("-" * 50)
    logger.info("⚠️ GESTION DES RISQUES (Niveau de confiance 95%)")
    logger.info(f"VaR 95% (Pire Sharpe)     : {var_95_sharpe:.3f} (Seuls 5% des scénarios font pire)")
    logger.info(f"VaR 95% (Pire Return)     : {var_95_return:.2f}%")
    logger.info(f"CVaR 95% (Crash moyen)    : {cvar_95_return:.2f}% (Moyenne des 5% pires scénarios)")
    logger.info(f"VaR 95% (Max Drawdown)    : {var_95_dd:.2f}%")
    logger.info("="*50)

    # --- 6. Visualisation des Distributions ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique 1 : Distribution du Sharpe
    sns.histplot(results_df['sharpe'], bins=15, kde=True, ax=axes[0], color='cyan')
    axes[0].axvline(results_df['sharpe'].mean(), color='yellow', linestyle='--', label=f"Moyenne: {results_df['sharpe'].mean():.2f}")
    axes[0].axvline(var_95_sharpe, color='red', linestyle='-', label=f"VaR 95%: {var_95_sharpe:.2f}")
    axes[0].set_title("Distribution du Ratio de Sharpe (Monte Carlo SDE)")
    axes[0].set_xlabel("Ratio de Sharpe")
    axes[0].legend()

    # Graphique 2 : Distribution du Max Drawdown
    sns.histplot(results_df['max_dd'], bins=15, kde=True, ax=axes[1], color='orange')
    axes[1].axvline(results_df['max_dd'].mean(), color='yellow', linestyle='--', label=f"Moyenne MDD: {results_df['max_dd'].mean():.2f}%")
    axes[1].axvline(var_95_dd, color='red', linestyle='-', label=f"VaR 95% MDD: {var_95_dd:.2f}%")
    axes[1].set_title("Distribution du Max Drawdown")
    axes[1].set_xlabel("Max Drawdown (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("monte_carlo_distribution.png", dpi=300)
    logger.info("📸 Graphique de distribution sauvegardé : 'monte_carlo_distribution.png'")

    return results_df

if __name__ == "__main__":
    run_monte_carlo_backtest()