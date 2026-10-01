import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import xgboost as xgb

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.backtest.generative_sde_engine import GenerativeSDEEngine
from research.backtest.engine import BacktestEngine, BacktestConfig
from research.signals.signal_generator import SignalGenerator, SignalGeneratorConfig
from features.features_eng import get_screener_feature_dict
from research.models.hmm_regimes.inference import RegimeDetector

def run_generative_cvar(pair_a, pair_b, n_paths=200, bars=2000):
    logger.info("=" * 60)
    logger.info(f"🧬 GÉNÉRATION CVaR & STRESS TEST MULTIVERS : {pair_a} × {pair_b}")
    logger.info("=" * 60)
    
    # 1. Load Parameters
    params_path = os.path.join(project_root, "data", "storage", "optimized_params", f"{pair_a}x{pair_b}_best_params.json")
    if not os.path.exists(params_path):
        logger.error(f"❌ Impossible de trouver {params_path}. Exécute d'abord run_full_pipeline.py !")
        return
        
    with open(params_path, "r") as f:
        best_params = json.load(f)
        
    logger.info(f"📄 Paramètres chargés : Z-Score={best_params['zscore_window']}, Entry={best_params['entry_threshold']}")
    
    # 2. Configurer les Moteurs
    bt_cfg = BacktestConfig(
        position_size=best_params["position_size"],
        stop_loss_z=best_params["stop_loss_z"],
        max_holding_bars=best_params["max_holding_bars"],
        cooldown_bars=best_params["cooldown_bars"]
    )
    sig_cfg = SignalGeneratorConfig(
        zscore_window=best_params["zscore_window"],
        entry_threshold=best_params["entry_threshold"],
        exit_threshold=best_params["exit_threshold"],
        compute_signature=False
    )
    
    logger.info("Initializing bt_engine")
    bt_engine = BacktestEngine(bt_cfg)
    logger.info("Initializing sig_gen")
    sig_gen = SignalGenerator(**sig_cfg.__dict__)
    
    # 3. Charger les Modèles ML (XGBoost et HMM)
    hmm_path = os.path.join(project_root, "data", "models", "hmm", "btc_regime_hmm.pkl")
    xgb_path = os.path.join(project_root, "data", "models", "meta_labeling", "universal_xgb_oracle.pkl")
    
    try:
        logger.info("Initializing HMM")
        hmm_oracle = RegimeDetector(hmm_path)
        logger.info("HMM success")
    except Exception as e:
        logger.error(f"HMM error: {e}")
        hmm_oracle = None
        
    try:
        logger.info("Initializing XGBoost")
        with open(xgb_path, "rb") as f:
            xgb_data = pickle.load(f)
        xgb_model = xgb_data["model"]
        xgb_features = xgb_data["features"]
        logger.info("XGBoost success")
    except Exception as e:
        logger.error(f"❌ Oracle XGBoost introuvable: {e}")
        return
        
    # 4. Préparer le contexte Macro (BTC / ETH pour les features XGB)
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_btc = pd.read_parquet(os.path.join(data_dir, "BTC_USDT.parquet")).set_index('timestamp')['close']
    df_eth = pd.read_parquet(os.path.join(data_dir, "ETH_USDT.parquet")).set_index('timestamp')['close']
    df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet")).set_index('timestamp')['close']
    df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet")).set_index('timestamp')['close']
    
    price_btc = df_btc.values[-bars:]
    price_eth = df_eth.values[-bars:]
    
    # 5. Generative SDE Paths
    # We dynamically look for the SDE .pt file for this pair
    models_dir = os.path.join(project_root, "data", "models")
    sde_models = [
        f for f in os.listdir(models_dir)
        if f.startswith("neural_sde_") and f.endswith("world_model.pt") 
    ]
    
    if not sde_models:
        logger.error(f"❌ Aucun modèle SDE trouvé pour {pair_a} / {pair_b}. Modèle attendu dans {models_dir}.")
        return
        
    sde_models.sort(reverse=True)
    sde_model_path = os.path.join(models_dir, sde_models[0])
    logger.info(f"Loading Neural SDE from {sde_model_path}...")
        
    synths_path = os.path.join(project_root, "data", "storage", "synthetic_paths", f"{pair_a}x{pair_b}_{n_paths}.parquet")
    
    p_a_init = df_a.iloc[-bars]
    p_b_init = df_b.iloc[-bars]
    
    if os.path.exists(synths_path):
        logger.info(f"📂 Chargement des trajectoires SDE CACHÉES depuis {synths_path}")
        df_paths = pd.read_parquet(synths_path)
        synthetic_markets = [group for _, group in df_paths.groupby(level=0)]
    else:
        logger.info(f"⏳ Génération de {n_paths} trajectoires mathématiques (SDE Multiverse)...")
        sde_engine = GenerativeSDEEngine(model_path=sde_model_path)
        synthetic_markets = sde_engine.simulate_markets(n_paths=n_paths, bars=bars, p_a_init=p_a_init, p_b_init=p_b_init)
        
        # Caching
        os.makedirs(os.path.dirname(synths_path), exist_ok=True)
        df_concat = pd.concat(synthetic_markets, keys=range(n_paths))
        df_concat.to_parquet(synths_path)
        logger.info(f"💾 Trajectoires SDE en Cache : {synths_path}")

    # 6. Evaluation Walk-Forward sur les univers synthétiques
    all_pure = []
    all_xgb = []
    
    log_btc = np.log(price_btc.astype(float))
    log_eth = np.log(price_eth.astype(float))
    
    for i, market_df in tqdm(enumerate(synthetic_markets), total=n_paths, desc="🧪 Stress-Test"):
        timestamps = np.arange(bars)
        price_a = market_df["SYNTH_A"].values.astype(float)
        price_b = market_df["SYNTH_B"].values.astype(float)
        
        # Absolute safety net to prevent SDE divergence crashes in Sklearn Walk-Forward
        price_a = np.nan_to_num(price_a, nan=100.0, posinf=100.0, neginf=100.0)
        price_b = np.nan_to_num(price_b, nan=100.0, posinf=100.0, neginf=100.0)
        price_a = np.clip(price_a, 1e-8, 1e8)
        price_b = np.clip(price_b, 1e-8, 1e8)
        
        # A. Stat ArB Pure
        signal = sig_gen.generate(timestamps, price_a, price_b, symbol_a=pair_a, symbol_b=pair_b)
        res_pure = bt_engine.run(signal, label=f"Path_{i}_PURE")
        metric_pure = res_pure["metrics"]
        all_pure.append(metric_pure)
        
        # B. XGBoost Oracle Filtered
        e_long = np.copy(signal.entry_long)
        e_short = np.copy(signal.entry_short)
        signal_indices = np.where((e_long == 1) | (e_short == 1))[0]
        
        log_a = np.log(price_a.astype(float))
        log_b = np.log(price_b.astype(float))
        zscores = signal.zscores
        
        for idx in signal_indices:
            h_start = max(0, idx - 127)
            # Feature extraction
            feat_dict = get_screener_feature_dict(
                log_a[h_start:idx+1], log_b[h_start:idx+1], log_btc[h_start:idx+1], log_eth[h_start:idx+1]
            )
            feat_dict["entry_z"] = float(zscores[idx])
            
            if hmm_oracle is not None:
                r_probs = hmm_oracle.predict_proba(log_btc[h_start:idx+1])
                feat_dict["hmm_regime_0"] = r_probs[0]
                feat_dict["hmm_regime_1"] = r_probs[1]
                feat_dict["hmm_regime_2"] = r_probs[2]
            else:
                feat_dict["hmm_regime_0"] = 0.0
                feat_dict["hmm_regime_1"] = 0.0
                feat_dict["hmm_regime_2"] = 0.0
                
            feat_vector = [feat_dict.get(c, 0.0) if np.isfinite(feat_dict.get(c, 0.0)) else 0.0 for c in xgb_features]
            x_arr = np.array(feat_vector, dtype=np.float32).reshape(1, -1)
            
            pred_prob = xgb_model.predict_proba(x_arr)[0][1]
            if pred_prob < 0.80:
                e_long[idx] = 0
                e_short[idx] = 0
                
        signal.entry_long = e_long
        signal.entry_short = e_short
        
        res_xgb = bt_engine.run(signal, label=f"Path_{i}_XGB")
        metric_xgb = res_xgb["metrics"]
        all_xgb.append(metric_xgb)
        
    # 7. Agrégation Statistique et Calculs CVaR (95%)
    df_pure = pd.DataFrame(all_pure)
    df_xgb = pd.DataFrame(all_xgb)
    
    var_95_ret_pure = np.percentile(df_pure['total_return_pct'], 5)
    cvar_95_ret_pure = df_pure[df_pure['total_return_pct'] <= var_95_ret_pure]['total_return_pct'].mean()
    
    var_95_ret_xgb = np.percentile(df_xgb['total_return_pct'], 5)
    cvar_95_ret_xgb = df_xgb[df_xgb['total_return_pct'] <= var_95_ret_xgb]['total_return_pct'].mean()
    
    var_95_dd_pure = np.percentile(df_pure['max_dd'], 5)
    var_95_dd_xgb = np.percentile(df_xgb['max_dd'], 5)
    
    logger.info("\n" + "="*70)
    logger.info(f"💎 RÉSULTATS GÉNÉRATIFS ({n_paths} MULTIVERS - CONFIANCE 95%)")
    logger.info("="*70)
    logger.info(f"{'MÉTRIQUE':<25} | {'STRATÉGIE PURE':<20} | {'XGBOOST ORACLE':<20}")
    logger.info("-"*70)
    logger.info(f"{'Sharpe Moyen':<25} | {df_pure['sharpe'].mean():<20.3f} | {df_xgb['sharpe'].mean():<20.3f}")
    logger.info(f"{'PnL Moyen (%)':<25} | {df_pure['total_return_pct'].mean():<20.2f} | {df_xgb['total_return_pct'].mean():<20.2f}")
    logger.info(f"{'Drawdown Moyen (%)':<25} | {df_pure['max_dd'].mean():<20.2f} | {df_xgb['max_dd'].mean():<20.2f}")
    logger.info("-"*70)
    logger.info(f"{'VaR 95% PnL (%)':<25} | {var_95_ret_pure:<20.2f} | {var_95_ret_xgb:<20.2f}")
    logger.info(f"{'CVaR 95% Crash Moyen (%)':<25} | {cvar_95_ret_pure:<20.2f} | {cvar_95_ret_xgb:<20.2f}")
    logger.info(f"{'VaR 95% Pire Drawdown (%)':<25} | {var_95_dd_pure:<20.2f} | {var_95_dd_xgb:<20.2f}")
    logger.info("="*70)

    # 8. Visuels
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.kdeplot(df_pure['total_return_pct'], ax=axes[0], color='red', label='Pure Stat Arb', fill=True, alpha=0.3)
    sns.kdeplot(df_xgb['total_return_pct'], ax=axes[0], color='cyan', label='XGBoost Oracle', fill=True, alpha=0.3)
    axes[0].set_title(f"Distribution Générative des PnL (%) - {pair_a}/{pair_b}")
    axes[0].axvline(0, color='white', linestyle='--', linewidth=1)
    axes[0].axvline(cvar_95_ret_pure, color='red', linestyle=':')
    axes[0].axvline(cvar_95_ret_xgb, color='cyan', linestyle=':')
    axes[0].legend()
    
    sns.kdeplot(df_pure['max_dd'], ax=axes[1], color='orange', label='Pure Stat Arb', fill=True, alpha=0.3)
    sns.kdeplot(df_xgb['max_dd'], ax=axes[1], color='magenta', label='XGBoost Oracle', fill=True, alpha=0.3)
    axes[1].set_title(f"Distribution Générative du Drawdown (%)")
    axes[1].axvline(var_95_dd_pure, color='orange', linestyle=':')
    axes[1].axvline(var_95_dd_xgb, color='magenta', linestyle=':')
    axes[1].legend()

    plot_path = os.path.join(project_root, "data", "storage", "synthetic_paths", f"{pair_a}x{pair_b}_cvar_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    logger.info(f"📸 Graphique de distribution sauvegardé : {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_a", type=str, default="ZEC_USDT")
    parser.add_argument("--pair_b", type=str, default="XRP_USDT")
    parser.add_argument("--n_paths", type=int, default=200)
    args = parser.parse_args()
    
    run_generative_cvar(args.pair_a, args.pair_b, args.n_paths)
