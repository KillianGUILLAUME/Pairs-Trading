import os
import sys
import pickle
import numpy as np
import pandas as pd
from loguru import logger
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.backtest.engine import BacktestEngine, BacktestConfig, PerformanceMetrics
from research.signals.signal_generator import SignalGenerator, SignalGeneratorConfig
from research.sizing.portfolio_manager import PortfolioManager, PortfolioConfig
from research.models.meta_labeling.xgb_walkforward import XGBWalkForwardEngine
from features.features_eng import get_screener_feature_dict
from research.models.hmm_regimes.inference import RegimeDetector

def load_data_matrix(pairs, data_dir):
    data = {}
    min_len = int(1e9)
    for pa, pb in pairs:
        df_a = pd.read_parquet(os.path.join(data_dir, f"{pa}.parquet"))
        df_b = pd.read_parquet(os.path.join(data_dir, f"{pb}.parquet"))
        data[pa] = df_a.set_index('timestamp')['close']
        data[pb] = df_b.set_index('timestamp')['close']
        min_len = min(min_len, len(df_a), len(df_b))
        
    df_btc = pd.read_parquet(os.path.join(data_dir, "BTC_USDT.parquet")).set_index('timestamp')['close']
    df_eth = pd.read_parquet(os.path.join(data_dir, "ETH_USDT.parquet")).set_index('timestamp')['close']
    data["BTC_USDT"] = df_btc
    data["ETH_USDT"] = df_eth
    
    df_mat = pd.DataFrame(data).dropna()
    return df_mat

def build_signals_and_probs(df_mat, pairs, xgb_model, xgb_features, hmm_oracle):
    timestamps = np.arange(len(df_mat))
    signals = {}
    probs_dict = {}
    
    log_btc = np.log(df_mat["BTC_USDT"].values.astype(float))
    log_eth = np.log(df_mat["ETH_USDT"].values.astype(float))
    
    for (pa, pb) in pairs:
        price_a = df_mat[pa].values.astype(float)
        price_b = df_mat[pb].values.astype(float)
        
        # Hardcoded robust parameters (ou charger depuis best_params.json)
        sig_cfg = SignalGeneratorConfig(zscore_window=156, entry_threshold=1.7, exit_threshold=0.35, compute_signature=False)
        sig_gen = SignalGenerator(**sig_cfg.__dict__)
        
        signal = sig_gen.generate(timestamps, price_a, price_b, symbol_a=pa, symbol_b=pb)
        signals[f"{pa}x{pb}"] = signal
        
        probs = np.zeros(len(df_mat))
        
        # Feature extraction for XGBoost
        e_long = signal.entry_long
        e_short = signal.entry_short
        trade_idx = np.where((e_long == 1) | (e_short == 1))[0]
        
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        zscores = signal.zscores
        
        for idx in trade_idx:
            h_start = max(0, idx - 127)
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
            probs[idx] = float(pred_prob)
            
        probs_dict[f"{pa}x{pb}"] = probs
        
    return signals, probs_dict

def run_portfolio_analysis():
    pairs = [("ZEC_USDT", "XRP_USDT"), ("ADA_USDT", "AVAX_USDT")]
    logger.info(f"🚀 DÉMARRAGE PORTFOLIO MANAGER TEST : {len(pairs)} PAIRES")
    
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_mat = load_data_matrix(pairs, data_dir)
    logger.info(f"📊 Matrice de données chargée : {df_mat.shape}")
    
    xgb_path = os.path.join(project_root, "data", "models", "meta_labeling", "universal_xgb_oracle.pkl")
    hmm_path = os.path.join(project_root, "data", "models", "hmm", "btc_regime_hmm.pkl")
    
    with open(xgb_path, "rb") as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data["model"]
    xgb_features = xgb_data["features"]
    
    try:
        hmm_oracle = RegimeDetector(hmm_path)
    except:
        hmm_oracle = None

    logger.info("🧠 Extraction des Signaux et Inférence XGBoost...")
    signals, probs_dict = build_signals_and_probs(df_mat, pairs, xgb_model, xgb_features, hmm_oracle)
    
    # Portfolio Allocation
    logger.info("💼 Exécution de l'Allocation Fractional Kelly Capping...")
    pm_cfg = PortfolioConfig(kelly_fraction=0.5, max_gross_exposure=2.0, max_pair_exposure=0.5)
    pm = PortfolioManager(pm_cfg)
    
    # We estimate historically Win_ret and Loss_ret roughly for these pairs.
    win_returns = {f"{pa}x{pb}": 0.02 for pa, pb in pairs}
    loss_returns = {f"{pa}x{pb}": 0.01 for pa, pb in pairs}
    
    allocations_df = pm.allocate_vectorized(probs_dict, win_returns, loss_returns)
    
    # Backtest execution
    bt_cfg = BacktestConfig(position_size=0.10, stop_loss_z=5.5, max_holding_bars=192, cooldown_bars=14)
    bt_engine = BacktestEngine(bt_cfg)
    
    portfolio_pnl = np.zeros(len(df_mat))
    
    for pair_name, signal in signals.items():
        # XGB Filter logic: Block entry if prob < 0.55
        prob_arr = probs_dict[pair_name]
        alloc_arr = allocations_df[pair_name].values
        
        signal.entry_long[prob_arr < 0.55] = 0
        signal.entry_short[prob_arr < 0.55] = 0
        
        # Override allocation purely where trades trigger
        res = bt_engine.run(signal, allocations=alloc_arr)
        pair_df = res["df"]
        portfolio_pnl += pair_df["net_pnl"].values
        
        logger.info(f"   ► [{pair_name}] Sharpe = {res['metrics']['sharpe']:.2f} | PnL = {res['metrics']['total_return_pct']:.2f}%")
        
    timestamps = df_mat.index
    total_capital = bt_cfg.initial_capital + np.cumsum(portfolio_pnl)
    
    # Simple Portfolio Metrics
    ret_port = np.diff(total_capital) / total_capital[:-1]
    sharpe_port = (ret_port.mean() / (ret_port.std() + 1e-9)) * np.sqrt(8760)
    max_dd_port = ((total_capital - np.maximum.accumulate(total_capital)) / np.maximum.accumulate(total_capital)).min()
    
    logger.info("=" * 60)
    logger.info("🌍 PORTFOLIO CONSOLIDÉ (FRACTIONAL KELLY & CAPPED EXPOSURE)")
    logger.info("=" * 60)
    logger.info(f"Max Gross Exposure Autorisé : {pm_cfg.max_gross_exposure*100}%")
    logger.info(f"Avg Gross Exposure Réalisé  : {allocations_df.sum(axis=1).mean()*100:.2f}%")
    logger.info(f"Sharpe Ratio                : {sharpe_port:.3f}")
    logger.info(f"Max Drawdown                : {max_dd_port*100:.2f}%")
    logger.info(f"Capital Final               : ${total_capital[-1]:.2f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_portfolio_analysis()
