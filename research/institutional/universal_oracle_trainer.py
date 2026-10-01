import os
import sys
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from loguru import logger
import concurrent.futures
from typing import Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.signal_engine import InstitutionalSignalEngine
from research.institutional.feature_master import FeatureMaster

def build_single_pair_dataset(pair_a: str, pair_b: str, window: int = 156):
    """
    Extrait les features et labels pour une seule paire.
    """
    try:
        data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
        df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
        df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
        
        df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        ts = df['timestamp'].values
        pa = df['close_a'].values.astype(float)
        pb = df['close_b'].values.astype(float)
        
        # Signaux Primaires (HJB)
        engine = InstitutionalSignalEngine(window=window, solver_type="fpt", cost=0.001)
        sig = engine.generate(ts, pa, pb)
        
        # Triple Barrier Labeling
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
        
        for idx in entries:
            if idx + timeout >= len(spreads): continue
            h_start = max(0, idx - 127)
            if idx - h_start < 64: continue
            
            f_dict = fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
            f_dict["spread_at_entry"] = spreads[idx]
            f_dict["dist_to_mu"] = np.abs(spreads[idx] - np.mean(spreads[h_start:idx+1]))
            
            # Labeling
            side = 1 if sig.entry_long[idx] == 1 else -1
            p_entry = spreads[idx]
            label = 0
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
            
        return np.array(X), np.array(y), list(f_dict.keys()) if X else None
    except Exception as e:
        logger.error(f"❌ Erreur sur {pair_a}x{pair_b} : {e}")
        return None, None, None

def train_universal_oracle():
    # 1. Sélection des paires
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    available_files = [f.replace(".parquet", "") for f in os.listdir(data_dir) if f.endswith(".parquet")]
    
    screened_path = os.path.join(project_root, "data", "storage", "screened/super_dataset_SDE_128.parquet")
    df_s = pd.read_parquet(screened_path)
    all_pairs = df_s[['pair_a', 'pair_b']].drop_duplicates().values.tolist()
    pairs = [(p[0], p[1]) for p in all_pairs if p[0] in available_files and p[1] in available_files]
    
    logger.info(f"🔨 CONSTRUCTION DU DATASET UNIVERSEL | {len(pairs)} PAIRES")
    
    X_agg = []
    y_agg = []
    feature_names = None
    
    # 2. Pipeline Parallèle
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(build_single_pair_dataset, p[0], p[1]): p for p in pairs}
        for future in concurrent.futures.as_completed(futures):
            X, y, f_names = future.result()
            if X is not None and len(X) > 0:
                X_agg.append(X)
                y_agg.append(y)
                if feature_names is None: feature_names = f_names
                logger.info(f"✅ Paire traitée : {futures[future]} | Samples: {len(X)}")

    X_final = np.concatenate(X_agg)
    y_final = np.concatenate(y_agg)
    
    logger.info(f"📊 Training Matrix: {X_final.shape} | Positive Class: {np.mean(y_final)*100:.1f}%")
    
    # 3. Training
    model = xgb.XGBClassifier(
        n_estimators=1000, # Augmenté pour capturer la richesse des 20+ features
        max_depth=8,       # Profondeur accrue pour les interactions stats/signatures
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist"  # Faster for large datasets
    )
    
    model.fit(X_final, y_final)
    
    # 4. Save
    out_dir = os.path.join(project_root, "data", "models", "institutional")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "oracle_v3_universal.pkl")
    
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "features": feature_names}, f)
        
    logger.info(f"🚀 Oracle V3 UNIVERSEL (Fused Features) sauvegardé dans {save_path}")

if __name__ == "__main__":
    train_universal_oracle()
