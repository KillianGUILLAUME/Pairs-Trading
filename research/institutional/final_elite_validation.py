import os
import sys
import numpy as np
import pandas as pd
import pickle
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.signal_engine import InstitutionalSignalEngine
from research.backtest.engine import BacktestEngine, BacktestConfig
from research.institutional.institutional_wfo import InstitutionalWFO

def run_elite_validation(pair_a: str, pair_b: str):
    logger.info(f"🧪 FINAL ELITE VALIDATION : {pair_a}/{pair_b}")
    
    # Best Params from Optuna
    elite_window = 220
    elite_cost = 0.0027
    elite_threshold = 0.60
    
    # 1. Load Data
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
    df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
    df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Setup Engine
    engine = InstitutionalSignalEngine(window=elite_window, cost=elite_cost, use_kalman=True)
    oracle_p = os.path.join(project_root, "data", "models", "institutional", "oracle_v3_universal.pkl")
    with open(oracle_p, "rb") as f:
        odata = pickle.load(f)
        oracle_model = odata["model"]

    # 3. Generate & Filter
    ts = df['timestamp'].values
    pa = df['close_a'].values
    pb = df['close_b'].values
    sig = engine.generate(ts, pa, pb)
    
    from research.institutional.feature_master import FeatureMaster
    fm = FeatureMaster()
    log_a, log_b = np.log(pa), np.log(pb)
    
    v_long, v_short = np.zeros(len(ts)), np.zeros(len(ts))
    for idx in np.where((sig.entry_long == 1) | (sig.entry_short == 1))[0]:
        h_start = max(0, idx - 127)
        if idx - h_start < 64: continue
        f_dict = fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
        f_dict["spread_at_entry"] = sig.spreads[idx]
        f_dict["dist_to_mu"] = np.abs(sig.spreads[idx] - np.mean(sig.spreads[h_start:idx+1]))
        prob = oracle_model.predict_proba(np.array([list(f_dict.values())]))[0, 1]
        
        if prob >= elite_threshold:
            if sig.entry_long[idx] == 1: v_long[idx] = 1
            else: v_short[idx] = 1
            
    sig.entry_long, sig.entry_short = v_long, v_short
    
    # 4. Out-of-Sample Backtest (Last 30%)
    split = int(len(ts) * 0.7)
    sig_test = InstitutionalWFO._slice_signal(None, sig, split, len(ts))
    
    bt_engine = BacktestEngine(BacktestConfig(initial_capital=100000, position_size=0.1))
    res = bt_engine.run(sig_test)
    
    metrics = res["metrics"]
    print("\n" + "="*50)
    print(f"🥇 ELITE OOS PERFORMANCE : {pair_a}/{pair_b}")
    print("="*50)
    print(f"Sharpe Ratio:  {metrics['sharpe']}")
    print(f"Total Return: {metrics['total_return_pct']}%")
    print(f"Max Drawdown: {metrics['max_dd']}%")
    print(f"Trades Count: {metrics['n_trades']}")
    print(f"Profit Factor: {metrics['profit_factor']}")

if __name__ == "__main__":
    run_elite_validation("ADA_USDT", "AVAX_USDT")
