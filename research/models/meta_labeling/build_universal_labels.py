import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from research.models.hmm_regimes.inference import RegimeDetector

def process_pair_group(group_name, subset_df, parquet_dir, hmm_path):
    pair_a, pair_b = group_name
    
    path_a = os.path.join(parquet_dir, f"{pair_a}.parquet")
    path_b = os.path.join(parquet_dir, f"{pair_b}.parquet")
    path_btc = os.path.join(parquet_dir, "BTC_USDT.parquet")
    
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return None
        
    df_a = pd.read_parquet(path_a).set_index('timestamp')['close']
    df_b = pd.read_parquet(path_b).set_index('timestamp')['close']
    df_btc = pd.read_parquet(path_btc).set_index('timestamp')['close']
    
    df_comb = pd.DataFrame({'a': df_a, 'b': df_b, 'btc': df_btc}).dropna()
    timestamps = df_comb.index.values
    log_a = np.log(df_comb['a'].values)
    log_b = np.log(df_comb['b'].values)
    log_btc = np.log(df_comb['btc'].values)
    
    ts_to_idx = {t: i for i, t in enumerate(timestamps)}
    
    # Instance légère du HMM pour ce process worker
    try:
        hmm_filter = RegimeDetector(hmm_path)
    except:
        hmm_filter = None
    
    results = []
    max_hold = 336
    sl_z = 4.0
    
    for _, row in subset_df.iterrows():
        t_end = row['timestamp_end']
        if t_end not in ts_to_idx:
            continue
            
        entry_idx = ts_to_idx[t_end]
        exit_idx = min(entry_idx + max_hold, len(log_a) - 1)
        
        w_a = log_a[entry_idx - 127: entry_idx + 1]
        w_b = log_b[entry_idx - 127: entry_idx + 1]
        
        # Si trop proche du début
        if len(w_a) < 128:
            continue
            
        b_cov = np.cov(w_a, w_b)[0, 1]
        a_var = np.var(w_a)
        
        if a_var == 0:
            continue
            
        beta = b_cov / a_var
        intercept = np.mean(w_b) - beta * np.mean(w_a)
        
        f_a = log_a[entry_idx : exit_idx + 1]
        f_b = log_b[entry_idx : exit_idx + 1]
        f_spread = f_b - beta * f_a - intercept
        
        entry_spread = f_spread[0]
        mu = 0.0 
        
        w_spread = w_b - beta * w_a - intercept
        sigma = np.std(w_spread)
        
        if sigma < 1e-6:
            continue
            
        z_entry = (entry_spread - mu) / sigma
        
        # PnL simulation (Short ou Long Spread vers la moyenne)
        position = -1 if entry_spread > 0 else 1
        
        f_pnl = position * (f_spread - entry_spread)
        f_z = (f_spread - mu) / sigma
        
        trade_win = 0
        for k in range(1, len(f_pnl)):
            z = f_z[k]
            pnl = f_pnl[k]
            
            if abs(z) >= sl_z:
                trade_win = 0
                break
                
            if position == -1 and z <= 0.2:
                trade_win = 1 if pnl > 0 else 0
                break
            if position == 1 and z >= -0.2:
                trade_win = 1 if pnl > 0 else 0
                break
                
        row_dict = row.to_dict()
        row_dict['target_y'] = trade_win
        row_dict['entry_z'] = z_entry
        
        if hmm_filter is not None:
            r_probs = hmm_filter.predict_proba(log_btc[entry_idx - 127: entry_idx + 1])
            row_dict['hmm_regime_0'] = r_probs[0]
            row_dict['hmm_regime_1'] = r_probs[1]
            row_dict['hmm_regime_2'] = r_probs[2]
        else:
            row_dict['hmm_regime_0'] = 0.0
            row_dict['hmm_regime_1'] = 0.0
            row_dict['hmm_regime_2'] = 0.0
            
        results.append(row_dict)
        
    return pd.DataFrame(results)

def main():
    screened_path = os.path.join(project_root, "data", "storage", "screened", "super_dataset_SDE_128.parquet")
    raw_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    hmm_path = os.path.join(project_root, "data", "models", "hmm", "btc_regime_hmm.pkl")
    out_path = os.path.join(project_root, "data", "models", "meta_labeling", "universal_dataset.parquet")
    
    logger.info("Lecture du super dataset 57k...")
    df_super = pd.read_parquet(screened_path)
    
    groups = list(df_super.groupby(['pair_a', 'pair_b']))
    logger.info(f"{len(groups)} paires uniques trouvées. Calcul des labels forward en MutiProcessing...")
    
    res_dfs = Parallel(n_jobs=-1)(
        delayed(process_pair_group)(name, subset, raw_dir, hmm_path)
        for name, subset in tqdm(groups, total=len(groups))
    )
    
    final_df = pd.concat([d for d in res_dfs if not d.empty], ignore_index=True)
    
    drop_cols = ['window_id', 'timestamp_start', 'timestamp_end', 'log_prices_a', 'log_prices_b', 'pair_a', 'pair_b']
    for c in drop_cols:
        if c in final_df.columns:
            final_df = final_df.drop(columns=[c])
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_parquet(out_path)
    
    logger.info(f"✅ Universal Labeling Terminé. Shape finale: {final_df.shape}")
    logger.info(f"💰 Distribution Gagnants (1) / Perdants (0) :\n{final_df['target_y'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()
