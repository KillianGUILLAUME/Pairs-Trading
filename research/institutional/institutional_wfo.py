import os
import sys
import numpy as np
import pandas as pd
import pickle
from loguru import logger
from dataclasses import dataclass
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.signal_engine import InstitutionalSignalEngine
from research.sizing.portfolio_manager import PortfolioManager, PortfolioConfig
from research.backtest.engine import BacktestEngine, BacktestConfig

class InstitutionalWFO:
    """
    Bridge entre la nouvelle plateforme institutionnelle et le legacy Backtester.
    """
    def __init__(self, oracle_path: str, threshold: float = 0.55):
        self.engine = InstitutionalSignalEngine(window=156, use_kalman=True)
        self.oracle_path = oracle_path
        self.threshold = threshold
        
        # Load Oracle
        with open(oracle_path, "rb") as f:
            data = pickle.load(f)
            self.oracle_model = data["model"]
            self.oracle_features = data["features"]
            
        from research.institutional.feature_master import FeatureMaster
        self.fm = FeatureMaster()

    def run_pair_wfo(self, pair_a: str, pair_b: str, train_ratio: float = 0.7):
        # 1. Load Data
        data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
        df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
        df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
        
        df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        ts = df['timestamp'].values
        pa = df['close_a'].values
        pb = df['close_b'].values
        
        n = len(ts)
        split_idx = int(n * train_ratio)
        
        # 2. Generate Signals (Full History for Kalman Warmup)
        sig = self.engine.generate(ts, pa, pb)
        
        # 3. Meta-Labeling (Universal Oracle V2)
        log_a = np.log(pa)
        log_b = np.log(pb)
        entries = np.where((sig.entry_long == 1) | (sig.entry_short == 1))[0]
        
        vetoed_long = np.zeros(n)
        vetoed_short = np.zeros(n)
        
        for idx in entries:
            h_start = max(0, idx - 127)
            if idx - h_start < 64: continue
            
            f_dict = self.fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
            f_dict["spread_at_entry"] = sig.spreads[idx]
            f_dict["dist_to_mu"] = np.abs(sig.spreads[idx] - np.mean(sig.spreads[h_start:idx+1]))
            
            vec = np.array([list(f_dict.values())])
            prob = self.oracle_model.predict_proba(vec)[0, 1]
            
            if prob >= self.threshold:
                if sig.entry_long[idx] == 1: vetoed_long[idx] = 1
                else: vetoed_short[idx] = 1
                
        # Remplace signals originaux par les signaux filtrés
        sig.entry_long = vetoed_long
        sig.entry_short = vetoed_short
        
        # 4. Legacy Backtest (Splits)
        bt_engine = BacktestEngine()
        
        # Train Split
        sig_train = self._slice_signal(sig, 0, split_idx)
        res_train = bt_engine.run(sig_train, label=f"{pair_a}/{pair_b} [TRAIN]")
        
        # Test Split
        sig_test = self._slice_signal(sig, split_idx, n)
        res_test = bt_engine.run(sig_test, label=f"{pair_a}/{pair_b} [TEST]")
        
        # 5. Stress Test (Base Scenario)
        from research.backtest.stress_test import StressTestEngine, FEE_SCENARIOS
        stress_engine = StressTestEngine()
        res_stress = stress_engine.run_scenario(sig_test, FEE_SCENARIOS["base"])
        
        return res_train, res_test, res_stress

    def _slice_signal(self, sig, start, end):
        # Helper to slice the InstitutionalSignal dataclass
        from research.institutional.signal_engine import InstitutionalSignal
        return InstitutionalSignal(
            timestamps=sig.timestamps[start:end],
            price_a=sig.price_a[start:end],
            price_b=sig.price_b[start:end],
            spreads=sig.spreads[start:end],
            zscores=sig.zscores[start:end],
            betas=sig.betas[start:end],
            entry_long=sig.entry_long[start:end],
            entry_short=sig.entry_short[start:end],
            exit_signal=sig.exit_signal[start:end],
            b_star=sig.b_star[start:end],
            d_star=sig.d_star[start:end]
        )

if __name__ == "__main__":
    oracle_p = os.path.join(project_root, "data", "models", "institutional", "oracle_v2_universal.pkl")
    wfo = InstitutionalWFO(oracle_p, threshold=0.55)
    
    # Test sur une paire représentative
    res_tr, res_te, res_st = wfo.run_pair_wfo("ADA_USDT", "AVAX_USDT")
    
    print("\n" + "="*50)
    print("🚀 INSTITUTIONAL VALIDATION (WFO & STRESS) : ADA/AVAX")
    print("="*50)
    print(f"TRAIN Sharpe: {res_tr['metrics']['sharpe']} | Trades: {res_tr['metrics']['n_trades']}")
    print(f"TEST  Sharpe: {res_te['metrics']['sharpe']} | Trades: {res_te['metrics']['n_trades']}")
    print(f"STRESS (Base) Sharpe: {res_st['metrics']['sharpe']}")
    print(f"STRESS (Base) Cost Drag: {res_st['metrics']['cost_drag_pct']}%")
