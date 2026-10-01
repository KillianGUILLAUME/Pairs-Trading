"""
Orchestrateur qui utilise screener.py pour scanner toutes les combinaisons
de paires disponibles, filtre les fenêtres cointégrées (pépites), 
et les compile dans un seul gros fichier Parquet pour le Neural SDE.
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# On importe ta fonction principale
from screener import run_screener

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "storage", "parquet")
DATA_DIR_1H = os.path.join(DATA_DIR, "1h")
OUTPUT_SUPER_DATASET = os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128.parquet")

WINDOW_SIZE = 128
STRIDE = 8

def get_available_symbols():
    """Lit les fichiers .parquet dans le dossier 1h pour trouver les cryptos."""
    files = glob.glob(os.path.join(DATA_DIR_1H, "*_USDT.parquet"))
    symbols = [os.path.basename(f).replace(".parquet", "") for f in files]
    # On filtre les stablecoins purs et le fiat si besoin
    exclude = ["USDC_USDT", "FDUSD_USDT", "EUR_USDT"]
    return sorted([s for s in symbols if s not in exclude])

def get_asset_clusters(symbols, data_dir, timeframe="1h"):
    """ Analyse l'historique complet pour clusteriser les tokens similaires via DBSCAN. """
    logger.info("🧠 Démarrage du Pre-Clustering (DBSCAN) pour limiter les paires...")
    
    series_list = []
    # 1. Chargement hyper-rapide (uniquement `close` et `timestamp`)
    for sym in tqdm(symbols, desc="Chargement des séries pour le clustering"):
        path = os.path.join(data_dir, timeframe, f"{sym}.parquet")
        if os.path.exists(path):
            df_sym = pd.read_parquet(path, columns=["timestamp", "close"])
            s = df_sym.set_index("timestamp")["close"].rename(sym)
            series_list.append(s)
            
    df_prices = pd.concat(series_list, axis=1)
    # L'intersection absolue de tous les actifs (on prend la fenêtre temporelle où tous coexistent)
    df_prices = df_prices.dropna()
    
    if len(df_prices) < 100:
        logger.warning("Historique commun trop court pour un clustering (<100 barres). Fallback en cluster global.")
        return {0: symbols}
        
    # 2. Log-returns et Corrélation
    returns = np.log(df_prices).diff().dropna()
    corr = returns.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    
    # 3. Distance de Pearson: d = sqrt(2*(1-rho)). 
    # Une corrélation de 1 donne d=0. Une corrélation de 0 donne d=1.41. 
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, 4.0))
    
    # 4. Clustering (eps = 1.05 --> correlation horaire minimale de ~ 0.45)
    # 0.8 était beaucoup trop stricts pour du 1h (qui contient beaucoup de bruit de microstructure)
    clustering = DBSCAN(eps=1.05, min_samples=2, metric="precomputed").fit(dist)
    labels = clustering.labels_
    
    clusters = {}
    for sym, label in zip(df_prices.columns, labels):
        if label != -1: # Ignorer le cluster -1 (les tokens "bruits" non corrélés)
            clusters.setdefault(label, []).append(sym)
            
    if not clusters:
        logger.warning("⚠️ Aucun cluster n'a émergé (corrélation trop faible sur ce marché). On force le scan de toutes les paires possibles.")
        return {0: symbols}
        
    n_kept = sum(len(c) for c in clusters.values())
    logger.info(f"🎯 Clustering terminé : {len(clusters)} clusters forts identifiés ! ({n_kept}/{len(symbols)} tokens retenus pertinents).")
    
    for c_id, members in clusters.items():
        logger.info(f"   Cluster {c_id} : {len(members)} actifs -> {', '.join(members)}")
        
    return clusters

def main():
    logger.info("🚀 Démarrage de la construction du Super-Dataset Deep Learning")
    
    # 1. Détecter les actifs
    symbols = get_available_symbols()
    logger.info(f"🪙 {len(symbols)} actifs trouvés dans le dossier 1h.")
    
    # 2. Clustering pour définir l'univers pertinent
    clusters = get_asset_clusters(symbols, DATA_DIR, timeframe="1h")
    
    # 3. Créer les combinaisons de paires INTRA-CLUSTER uniquement
    all_pairs = []
    for c_id, members in clusters.items():
        if len(members) >= 2:
            all_pairs.extend(list(itertools.combinations(members, 2)))
            
    # Si on tombe à 0 paires (très peu probable), sécurité :
    if not all_pairs:
        all_pairs = list(itertools.combinations(symbols, 2))
        
    logger.info(f"🔀 {len(all_pairs)} combinaisons de paires à scanner (filtrées par clustering DBSCAN).")
    
    super_dataset_list = []
    
    # 3. Boucle sur les paires (La parallélisation interne se fait dans run_screener)
    for pair_a, pair_b in tqdm(all_pairs, desc="Analyse des paires"):
        try:
            # On appelle ton screener (qui sauvegarde aussi son propre historique en passant)
            # On met output_dir dans un sous-dossier temp pour ne pas polluer si tu veux
            df_screened = run_screener(
                pair_a=pair_a, 
                pair_b=pair_b, 
                timeframe="1h",
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                n_jobs=max(1, os.cpu_count() // 2),
                data_dir=os.path.join(PROJECT_ROOT, "data", "storage", "parquet"),
                output_dir=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "raw_pairs")
            )
            
            if df_screened is not None and not df_screened.empty:
                # 4. LE FILTRE D'OR : On ne garde que les pépites stationnaires pour l'IA
                gold_windows = df_screened[df_screened["adf_pvalue"] < 0.15].copy()
                
                if not gold_windows.empty:
                    # On ajoute l'identité pour le DataLoader
                    gold_windows["pair_a"] = pair_a
                    gold_windows["pair_b"] = pair_b
                    
                    super_dataset_list.append(gold_windows)
                    
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {pair_a} × {pair_b} : {e}")
            continue

    # 5. Agrégation finale
    if not super_dataset_list:
        logger.error("❌ Aucune fenêtre cointégrée trouvée sur tout l'univers. C'est anormal.")
        return

    logger.info("🧩 Fusion des données en cours...")
    df_super = pd.concat(super_dataset_list, ignore_index=True)
    df_super = df_super.sort_values("timestamp_start").reset_index(drop=True)
    
    # 6. Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_SUPER_DATASET), exist_ok=True)
    df_super.to_parquet(OUTPUT_SUPER_DATASET, engine="pyarrow", index=False)
    
    size_mb = os.path.getsize(OUTPUT_SUPER_DATASET) / (1024 * 1024)
    
    print("\n" + "="*60)
    print("🏆 SUPER-DATASET GÉNÉRÉ AVEC SUCCÈS")
    print("="*60)
    print(f"   Paires scannées         : {len(all_pairs)}")
    print(f"   Total Fenêtres 'Gold'   : {len(df_super)}")
    print(f"   Poids du fichier        : {size_mb:.1f} MB")
    print(f"   Chemin                  : {OUTPUT_SUPER_DATASET}")
    print("="*60)

if __name__ == "__main__":
    main()