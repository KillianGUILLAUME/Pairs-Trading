import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research.optimization.optimize_signals import WFOConfig, walk_forward_splits, run_optimization
from research.backtest.engine import BacktestConfig
from research.signals.signal_generator import SignalGeneratorConfig
from research.models.meta_labeling.xgb_walkforward import XGBWalkForwardEngine

import json

def run_global_pipeline(pair_a, pair_b, fee_scenario="base", num_trials=5):
    logger.info(f"🚀 INITIALISATION PIPELINE GLOBAL: {pair_a} × {pair_b}")
    
    # 1. OPTUNA - Bayesian Global Search
    logger.info("🛠 ÉTAPE 1: Optimisation Bayésienne des hyperparamètres (Signal PuR)...")
    study = run_optimization(pair_a=pair_a, pair_b=pair_b, n_trials=num_trials, n_folds=5, fee_scenario=fee_scenario)
    
    best_params = study.best_trial.params
    
    # 💾 SAUVEGARDE DES PARAMÈTRES POUR LE SDE GENERATIVE CVAR
    params_dir = os.path.join(project_root, "data", "storage", "optimized_params")
    os.makedirs(params_dir, exist_ok=True)
    params_path = os.path.join(params_dir, f"{pair_a}x{pair_b}_best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ Paramètres Optuna sauvegardés : {params_path}")
    
    # 2. XGBoost - Meta-Label Initialization
    logger.info("🧠 ÉTAPE 2: Walk-Forward Dynamique avec Model XGBoost Filter...")
    
    bt_cfg = BacktestConfig(
        position_size=best_params["position_size"],
        entry_threshold=best_params["entry_threshold"],
        exit_threshold=best_params["exit_threshold"],
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
    
    hmm_path = os.path.join(project_root, "data", "models", "hmm", "btc_regime_hmm.pkl")
    xgb_engine = XGBWalkForwardEngine(hmm_path=hmm_path, backtest_config=bt_cfg, sig_config=sig_cfg)
    
    # CHARGEMENT DONNÉES
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
    df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
    df_btc = pd.read_parquet(os.path.join(data_dir, "BTC_USDT.parquet"))
    df_eth = pd.read_parquet(os.path.join(data_dir, "ETH_USDT.parquet"))
    
    df = pd.DataFrame({
        'A': df_a.set_index('timestamp')['close'],
        'B': df_b.set_index('timestamp')['close'],
        'BTC': df_btc.set_index('timestamp')['close'],
        'ETH': df_eth.set_index('timestamp')['close'],
    }).dropna()
    
    ts = df.index.values
    price_a = df["A"].values
    price_b = df["B"].values
    price_btc = df["BTC"].values
    price_eth = df["ETH"].values
    
    wfo_config = WFOConfig(n_folds=5)
    splits = walk_forward_splits(len(price_a), wfo_config)
    
    stats_pure = []
    stats_xgb = []
    
    for i, ((train_s, train_e), (test_s, test_e)) in enumerate(splits):
        logger.info(f"🔄 FOLD WFO {i+1}/5 | Train: [{train_s}:{train_e}] → Test: [{test_s}:{test_e}]")
        
        sub_ts = ts[:test_e]
        sub_pa = price_a[:test_e]
        sub_pb = price_b[:test_e]
        sub_btc = price_btc[:test_e]
        sub_eth = price_eth[:test_e]
        
        ratio = train_e / test_e
        xgb_engine.wf_cfg.train_ratio = ratio
        xgb_engine.wf_cfg.min_bars = 500
        
        res = xgb_engine.run_pair_with_xgb(
            sub_ts, sub_pa, sub_pb, sub_btc, sub_eth,
            pair_a.replace("_USDT", ""), pair_b.replace("_USDT", "")
        )
        
        if res:
            m_pure = res["test_pure"]["metrics"]
            m_xgb = res["test_xgb"]["metrics"]
            
            stats_pure.append(m_pure)
            stats_xgb.append(m_xgb)
            
            logger.info(f"   ► PURE Sharpe : {m_pure['sharpe']:.3f} | PnL : {m_pure['total_return_pct']:.2f}% | Drawdown : {m_pure['max_dd']:.2f}%")
            logger.info(f"   ► XGB  Sharpe : {m_xgb['sharpe']:.3f} | PnL : {m_xgb['total_return_pct']:.2f}% | Drawdown : {m_xgb['max_dd']:.2f}%")
            
    # MOYENNES FINALES
    avg_sharpe_pure = np.mean([s['sharpe'] for s in stats_pure])
    avg_pnl_pure = np.mean([s['total_return_pct'] for s in stats_pure])
    avg_dd_pure = np.mean([s['max_dd'] for s in stats_pure])
    
    avg_sharpe_xgb = np.mean([s['sharpe'] for s in stats_xgb])
    avg_pnl_xgb = np.mean([s['total_return_pct'] for s in stats_xgb])
    avg_dd_xgb = np.mean([s['max_dd'] for s in stats_xgb])
    
    print("\n" + "="*80)
    print("💎 BILAN FINAL WALK-FORWARD (Moyenne sur les 5 Folds Non-Vus)")
    print("="*80)
    print(f"{'METRIQUE':<20} | {'STAT ARB (OPTUNA ONLY)':<25} | {'STAT ARB + XGBOOST ORACLE':<25}")
    print("-"*80)
    print(f"{'Sharpe Ratio':<20} | {avg_sharpe_pure:<25.3f} | {avg_sharpe_xgb:<25.3f}")
    print(f"{'Return (PnL %)':<20} | {avg_pnl_pure:<25.2f}% | {avg_pnl_xgb:<25.2f}%")
    print(f"{'Max Drawdown (%)':<20} | {avg_dd_pure:<25.2f}% | {avg_dd_xgb:<25.2f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_a", type=str, default="ZEC_USDT")
    parser.add_argument("--pair_b", type=str, default="XRP_USDT")
    args = parser.parse_args()
    
    run_global_pipeline(args.pair_a, args.pair_b)
