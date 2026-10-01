import os
import sys
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from loguru import logger
from typing import Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.signal_engine import InstitutionalSignalEngine
from research.institutional.feature_master import FeatureMaster
from research.backtest.engine import BacktestEngine, BacktestConfig

def build_institutional_dataset(pair_a: str, pair_b: str, window: int = 156):
    logger.info(f"🔨 CONSTRUCTION DU DATASET V2 : {pair_a} × {pair_b}")
    
    # 1. Chargement
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
    df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
    
    df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    ts = df['timestamp'].values
    pa = df['close_a'].values.astype(float)
    pb = df['close_b'].values.astype(float)
    
    # 2. Signaux Primaires (HJB)
    engine = InstitutionalSignalEngine(window=window, solver_type="fpt", cost=0.001)
    sig = engine.generate(ts, pa, pb)
    
    # 3. Triple Barrier Labeling
    # On définit les barrières (ex: 2% profit, 1% loss, 120h timeout)
    target_pct = 0.02
    stop_pct = 0.015
    timeout = 168 # 1 semaine
    
    log_a = np.log(pa)
    log_b = np.log(pb)
    spreads = sig.spreads
    
    entries = np.where((sig.entry_long == 1) | (sig.entry_short == 1))[0]
    
    X = []
    y = []
    
    fm = FeatureMaster(signature_level=2)
    
    logger.info(f"🧬 Extraction des features et Labélisation de {len(entries)} trades...")
    
    for idx in entries:
        if idx + timeout >= len(spreads):
            continue
            
        # --- Features (Window de 128 avant l'entrée) ---
        h_start = max(0, idx - 127)
        if idx - h_start < 64: continue
        
        f_dict = fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
        # On ajoute des features d'état OU
        f_dict["spread_at_entry"] = spreads[idx]
        f_dict["dist_to_mu"] = np.abs(spreads[idx] - np.mean(spreads[h_start:idx+1]))
        
        # --- Labeling (Forward look) ---
        side = 1 if sig.entry_long[idx] == 1 else -1
        p_entry = spreads[idx]
        
        label = 0 # Perdant par défaut
        for t in range(1, timeout):
            p_now = spreads[idx + t]
            ret = (p_now - p_entry) if side == 1 else (p_entry - p_now)
            
            if ret >= target_pct:
                label = 1
                break
            if ret <= -stop_pct:
                label = 0
                break
        
        X.append(list(f_dict.values()))
        y.append(label)
        
    feature_names = list(f_dict.keys())
    return np.array(X), np.array(y), feature_names

def train_institutional_oracle():
    # On entraîne sur ADA/AVAX et on peut ajouter d'autres paires pour la robustesse
    X, y, features = build_institutional_dataset("ADA_USDT", "AVAX_USDT")
    
    logger.info(f"📊 Training Matrix: {X.shape} | Positive Class: {np.mean(y)*100:.1f}%")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    
    model.fit(X, y)
    
    # Save
    out_dir = os.path.join(project_root, "data", "models", "institutional")
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, "oracle_v2_ada_avax.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
        
    logger.info(f"🚀 Oracle V2 sauvegardé dans {save_path}")

if __name__ == "__main__":
    train_institutional_oracle()
