import os
import sys
import numpy as np
import pandas as pd
import pickle
import optuna
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.signal_engine import InstitutionalSignalEngine
from research.backtest.engine import BacktestEngine, BacktestConfig
from research.institutional.feature_master import FeatureMaster

class InstitutionalOptimizer:
    def __init__(self, pair_a: str, pair_b: str, oracle_path: str):
        self.pair_a = pair_a
        self.pair_b = pair_b
        self.oracle_path = oracle_path
        
        # Load Data once
        data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
        df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
        df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
        df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
        self.df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Load Oracle
        with open(oracle_path, "rb") as f:
            odata = pickle.load(f)
            self.oracle_model = odata["model"]
            self.oracle_features = odata["features"]
            
        self.fm = FeatureMaster()

    def objective(self, trial):
        # 1. Hyperparams to optimize
        hjb_window = trial.suggest_int("hjb_window", 100, 300, step=20)
        hjb_cost = trial.suggest_float("hjb_cost", 0.0005, 0.005)
        oracle_threshold = trial.suggest_float("oracle_threshold", 0.50, 0.60)
        
        # 2. Run Signal Engine
        engine = InstitutionalSignalEngine(window=hjb_window, cost=hjb_cost, use_kalman=True)
        ts = self.df['timestamp'].values
        pa = self.df['close_a'].values
        pb = self.df['close_b'].values
        
        sig = engine.generate(ts, pa, pb)
        
        # 3. Apply Oracle Filter
        log_a = np.log(pa)
        log_b = np.log(pb)
        entries = np.where((sig.entry_long == 1) | (sig.entry_short == 1))[0]
        
        v_long = np.zeros(len(ts))
        v_short = np.zeros(len(ts))
        
        for idx in entries:
            h_start = max(0, idx - 127)
            if idx - h_start < 64: continue
            
            f_dict = self.fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
            f_dict["spread_at_entry"] = sig.spreads[idx]
            f_dict["dist_to_mu"] = np.abs(sig.spreads[idx] - np.mean(sig.spreads[h_start:idx+1]))
            
            vec = np.array([list(f_dict.values())])
            prob = self.oracle_model.predict_proba(vec)[0, 1]
            
            if prob >= oracle_threshold:
                if sig.entry_long[idx] == 1: v_long[idx] = 1
                else: v_short[idx] = 1
        
        sig.entry_long = v_long
        sig.entry_short = v_short
        
        # 4. Backtest (Train Split focus)
        split = int(len(ts) * 0.7)
        # Slicing signal manually
        from research.institutional.institutional_wfo import InstitutionalWFO
        sig_train = InstitutionalWFO._slice_signal(None, sig, 0, split)
        
        bt_engine = BacktestEngine(BacktestConfig(initial_capital=100000, position_size=0.1))
        res = bt_engine.run(sig_train)
        
        return res["metrics"]["sharpe"]

    def run_optimization(self, n_trials=50):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"🏆 BEST PARAMS : {study.best_params}")
        logger.info(f"📈 BEST SHARPE : {study.best_value}")
        
        # Sauvegarde des meilleurs paramètres
        out_path = os.path.join(project_root, "data", "models", "institutional", f"elite_params_{self.pair_a}_{self.pair_b}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(study.best_params, f)
            
        return study.best_params

if __name__ == "__main__":
    oracle_p = os.path.join(project_root, "data", "models", "institutional", "oracle_v2_universal.pkl")
    opt = InstitutionalOptimizer("ADA_USDT", "AVAX_USDT", oracle_p)
    opt.run_optimization(n_trials=30)
