"""
update_features.py

Script utilitaire autonome (Retro-Computing).
Permet d'appliquer de nouvelles features (développées dans features_eng.py)
sur des datasets existants SANS avoir à relancer le screening temporel 
ou les tests de cointégration (ADF/Hurst).

Il lit les fenêtres glissantes "log_prices" directement depuis le Parquet,
et met à jour les colonnes manquantes.
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from features.features_eng import get_screener_feature_dict

def process_file(file_path):
    print(f"⚙️ Scan de {os.path.basename(file_path)}...")
    try:
        # Lecture (fastparquet pour parer aux issues ARM64 cross-platform)
        df = pd.read_parquet(file_path, engine="fastparquet")
        
        # S'il est vide, on skippe
        if df.empty:
            return 0
            
        # Exemple de log_prices
        # Pour extraire les colonnes qu'on attend de features_eng
        # On va calculer sur la première ligne pour voir quelles sont les colonnes "cibles"
        row0 = df.iloc[0]
        log_a = np.array(row0["log_prices_a"], dtype=float)
        log_b = np.array(row0["log_prices_b"], dtype=float)
        
        # Btc et Eth peuvent manquer ou être dans le dict 
        # (S'il n'y a pas de btc_full dans le row on approxime avec des zéros si pas dispo, 
        # mais on n'update PAS les features macro btc_return ici)
        # Mais dans notre cas, log_btc et log_eth ne sont pas sauvegardées (pour gagner de la place).
        # Les features macro ont déjà été calculées. On Retro-compute *surtout* les features Quant MLE (skew, jump, student).
        # Si une feature nécessite le BTC/ETH, il faudrait faire un inner join avec le marché. 
        # Pour rester autonome et asynchrone, on fournit juste les vecteurs A et B purs.
        
        # /!\ ATTENTION: Etant donné que la fonction get_screener_feature_dict demande btc et eth,
        # on lui passe des mock arrays vides s'ils ne sont pas strictements liés aux paires A et B.
        mock_macro = np.zeros_like(log_a) 
        
        target_features = get_screener_feature_dict(log_a, log_b, mock_macro, mock_macro)
        
        missing_cols = [col for col in target_features.keys() if col not in df.columns]
        
        if not missing_cols:
            return 0 # Déjà à jour !!
            
        print(f"🔧 {os.path.basename(file_path)} (Manque : {missing_cols})")
        
        # On calcule les nouvelles features ligne par ligne
        # /!\ On ne recalcule QUE s'il manque des colonnes pour de l'optimisation
        
        def compute_missing(row):
            la = np.array(row["log_prices_a"], dtype=float)
            lb = np.array(row["log_prices_b"], dtype=float)
            # Re-calcule tout (c'est très rapide en vectoriel pur sans windowing)
            feat = get_screener_feature_dict(la, lb, mock_macro, mock_macro)
            return pd.Series({k: feat[k] for k in missing_cols})
            
        new_cols_df = df.apply(compute_missing, axis=1)
        
        # Fusion et sauvegarde
        for col in missing_cols:
            df[col] = new_cols_df[col]
            
        df.to_parquet(file_path, engine="pyarrow", index=False)
        return len(df)
        
    except Exception as e:
        print(f"❌ Erreur sur {file_path} : {e}")
        return 0

def main():
    raw_dir = os.path.join(project_root, "data", "storage", "screened", "raw_pairs")
    files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    
    if not files:
        print("Aucun fichier Parquet trouvé dans raw_pairs.")
        return
        
    print(f"🚀 Début du Retro-Computing sur {len(files)} paires...")
    t0 = time.time()
    
    # Parallélisation du patch
    results = Parallel(n_jobs=max(1, os.cpu_count() // 2), verbose=5)(
        delayed(process_file)(f) for f in files
    )
    
    total_updated = sum(results)
    print(f"\n==========================================")
    print(f"✅ RETRO-COMPUTING TERMINÉ ({time.time()-t0:.1f}s)")
    print(f"   -> {total_updated} fenêtres ont reçu les nouvelles features.")
    print(f"==========================================")
    
    if total_updated > 0:
        print("\n💡 Pense à relancer `python research/pairs/build_screened_dataset.py`")
        print("pour reconstruire le super_dataset_SDE.parquet final avec ces nouvelles colonnes !")

if __name__ == "__main__":
    main()
