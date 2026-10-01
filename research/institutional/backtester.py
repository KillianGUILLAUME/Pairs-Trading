import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from loguru import logger
import argparse
from research.institutional.feature_master import FeatureMaster
from research.backtest.engine import BacktestEngine, BacktestConfig, PerformanceMetrics
from research.institutional.signal_engine import InstitutionalSignalEngine
from research.institutional.funding_estimator import FundingEstimator
from visualization.plot.spread import plot_spread_diagnostic

def run_institutional_comparison(pair_a: str, pair_b: str):
    logger.info(f"📍 ANALYSE INSTITUTIONNELLE : {pair_a} × {pair_b}")
    
    # 1. Chargement des données 1h
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
    df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
    
    # Merge on timestamp
    df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    ts = df['timestamp'].values
    pa = df['close_a'].values.astype(float)
    pb = df['close_b'].values.astype(float)
    
    # 2. Estimation du Funding
    fe = FundingEstimator()
    funding_a = fe.estimate_funding_series(pa)
    funding_b = fe.estimate_funding_series(pb)
    
    # 3. Chargement de l'Oracle V2
    oracle_path = os.path.join(project_root, "data", "models", "institutional", "oracle_v2_ada_avax.pkl")
    with open(oracle_path, "rb") as f:
        oracle_data = pickle.load(f)
    oracle_model = oracle_data["model"]
    oracle_features = oracle_data["features"]
    fm = FeatureMaster(signature_level=2)
    log_a, log_b = np.log(pa), np.log(pb)

    # 4. Exécution des deux Solveurs HJB
    results = {}
    for solver_name in ["fpt", "dp"]:
        logger.info(f"🧠 Calibrage & Résolution HJB ({solver_name.upper()}) + VETO ORACLE...")
        engine = InstitutionalSignalEngine(window=156, solver_type=solver_name, cost=0.001)
        signal = engine.generate(ts, pa, pb)
        
        # Application du Veto Oracle
        entries = np.where((signal.entry_long == 1) | (signal.entry_short == 1))[0]
        veto_count = 0
        
        for idx in entries:
            h_start = max(0, idx - 127)
            if idx - h_start < 64: 
                signal.entry_long[idx] = 0
                signal.entry_short[idx] = 0
                continue
            
            f_dict = fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
            f_dict["spread_at_entry"] = signal.spreads[idx]
            f_dict["dist_to_mu"] = np.abs(signal.spreads[idx] - np.mean(signal.spreads[h_start:idx+1]))
            
            # Check feature alignment
            x_vec = [f_dict.get(fname, 0.0) for fname in oracle_features]
            prob = oracle_model.predict_proba(np.array([x_vec]))[0][1]
            
            if prob < 0.60:
                signal.entry_long[idx] = 0
                signal.entry_short[idx] = 0
                veto_count += 1
                
        logger.info(f"🚫 Oracle Veto : {veto_count} trades rejetés sur {len(entries)}")
        
        # Backtest
        bt_cfg = BacktestConfig(initial_capital=100000, position_size=0.10)
        bt = BacktestEngine(bt_cfg)
        bt_res = bt.run(signal)
        df_pnl = bt_res["df"]
        
        # Application du carry
        pos = df_pnl["position"].values
        # Carry = -pos * fund_a + pos * fund_b (approximativement)
        carry_pct = -pos * funding_a + pos * funding_b
        df_pnl["net_pnl"] = df_pnl["net_pnl"] + (carry_pct * df_pnl["capital"].shift(1).fillna(100000))
        
        # Recompute metrics after carry
        metrics = PerformanceMetrics.compute(df_pnl)
        results[solver_name] = {"metrics": metrics, "signal": signal}
        
        logger.info(f"✅ {solver_name.upper()} : Sharpe={metrics['sharpe']} | PnL={metrics['total_return_pct']}%")

    # 4. Comparaison Finale
    logger.info("="*50)
    logger.info("RÉSUMÉ COMPARATIF :")
    for s, data in results.items():
        m = data["metrics"]
        logger.info(f"► {s.upper():3s} | Sharpe: {m['sharpe']:6.3f} | MDD: {m['max_dd']:6.2f}% | Final: ${m['final_capital']}")
    logger.info("="*50)

if __name__ == "__main__":
    run_institutional_comparison("ADA_USDT", "AVAX_USDT")
