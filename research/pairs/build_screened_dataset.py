"""
Orchestrateur : scanne les paires intra-cluster (DBSCAN sur les corrélations de
rendements) avec le screener CAUSAL, garde les fenêtres cibles stationnaires et
les compile dans un seul Parquet pour la Neural SDE.

Sortie : data/storage/screened/super_dataset_SDE_128_causal.parquet
         + super_dataset_SDE_128_causal_meta.json (paramètres du screener)

Usage :
    python research/pairs/build_screened_dataset.py            # toutes les paires
    python research/pairs/build_screened_dataset.py --limit 3  # test rapide
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import itertools
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screener import run_screener  # noqa: E402

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "storage", "parquet")
DATA_DIR_1H = os.path.join(DATA_DIR, "1h")
SCREENED_DIR = os.path.join(PROJECT_ROOT, "data", "storage", "screened")
OUTPUT_SUPER_DATASET = os.path.join(SCREENED_DIR, "super_dataset_SDE_128_causal.parquet")

WINDOW_SIZE = 128
PAST_WINDOW = 128
STRIDE = 8
GOLD_ADF = 0.15


def get_available_symbols():
    files = glob.glob(os.path.join(DATA_DIR_1H, "*_USDT.parquet"))
    symbols = [os.path.basename(f).replace(".parquet", "") for f in files]
    exclude = ["USDC_USDT", "FDUSD_USDT", "EUR_USDT"]
    return sorted([s for s in symbols if s not in exclude])


def get_asset_clusters(symbols, data_dir, timeframe="1h"):
    logger.info("🧠 Pre-Clustering (DBSCAN) pour limiter les paires...")
    series_list = []
    for sym in tqdm(symbols, desc="Chargement des séries pour le clustering"):
        path = os.path.join(data_dir, timeframe, f"{sym}.parquet")
        if os.path.exists(path):
            df_sym = pd.read_parquet(path, columns=["timestamp", "close"])
            series_list.append(df_sym.set_index("timestamp")["close"].rename(sym))

    df_prices = pd.concat(series_list, axis=1).dropna()
    if len(df_prices) < 100:
        logger.warning("Historique commun trop court (<100 barres). Fallback en cluster global.")
        return {0: symbols}

    returns = np.log(df_prices).diff().dropna()
    corr = np.nan_to_num(returns.corr().values, nan=0.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, 4.0))
    labels = DBSCAN(eps=1.05, min_samples=2, metric="precomputed").fit(dist).labels_

    clusters = {}
    for sym, label in zip(df_prices.columns, labels):
        if label != -1:
            clusters.setdefault(label, []).append(sym)
    if not clusters:
        logger.warning("⚠️ Aucun cluster : scan de toutes les paires.")
        return {0: symbols}

    n_kept = sum(len(c) for c in clusters.values())
    logger.info(f"🎯 {len(clusters)} clusters ({n_kept}/{len(symbols)} tokens).")
    for c_id, members in clusters.items():
        logger.info(f"   Cluster {c_id} : {', '.join(members)}")
    return clusters


def main(limit: int | None = None, n_jobs: int | None = None, fast_features: bool = True):
    logger.info("🚀 Construction du super-dataset CAUSAL pour la Neural SDE")
    symbols = get_available_symbols()
    logger.info(f"🪙 {len(symbols)} actifs trouvés dans le dossier 1h.")

    clusters = get_asset_clusters(symbols, DATA_DIR, timeframe="1h")
    all_pairs = []
    for members in clusters.values():
        if len(members) >= 2:
            all_pairs.extend(itertools.combinations(members, 2))
    if not all_pairs:
        all_pairs = list(itertools.combinations(symbols, 2))
    if limit:
        all_pairs = all_pairs[:limit]
    logger.info(f"🔀 {len(all_pairs)} paires à scanner.")

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 2)
    raw_dir = os.path.join(SCREENED_DIR, "raw_pairs_causal")
    super_dataset_list = []

    for pair_a, pair_b in tqdm(all_pairs, desc="Analyse des paires"):
        try:
            df_screened = run_screener(
                pair_a=pair_a, pair_b=pair_b, timeframe="1h",
                window_size=WINDOW_SIZE, past_window=PAST_WINDOW, stride=STRIDE,
                n_jobs=n_jobs, data_dir=DATA_DIR, output_dir=raw_dir,
                fast_features=fast_features, verbose=0,
            )
            if df_screened is not None and not df_screened.empty:
                gold = df_screened[df_screened["target_adf_pvalue"] < GOLD_ADF].copy()
                if not gold.empty:
                    gold["pair_a"] = pair_a
                    gold["pair_b"] = pair_b
                    super_dataset_list.append(gold)
        except Exception as e:
            logger.error(f"Erreur sur {pair_a} × {pair_b} : {e!r}")
            continue

    if not super_dataset_list:
        logger.error("❌ Aucune fenêtre cointégrée trouvée.")
        return

    df_super = pd.concat(super_dataset_list, ignore_index=True)
    df_super = df_super.sort_values("timestamp_start").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_SUPER_DATASET), exist_ok=True)
    df_super.to_parquet(OUTPUT_SUPER_DATASET, engine="pyarrow", index=False)
    import hashlib
    def _sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    raw_hashes = {s: _sha(os.path.join(DATA_DIR_1H, f"{s}.parquet")) for s in symbols}
    meta = {
        "schema": "causal_v1",
        "dataset_sha256": _sha(OUTPUT_SUPER_DATASET),
        "raw_inputs_sha256": raw_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode().strip() if os.path.isdir(os.path.join(PROJECT_ROOT, ".git")) else None,
        "window_size": WINDOW_SIZE, "past_window": PAST_WINDOW, "stride": STRIDE,
        "gold_adf_threshold": GOLD_ADF, "fast_features": fast_features,
        "n_pairs": len(all_pairs), "n_windows": int(len(df_super)),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.splitext(OUTPUT_SUPER_DATASET)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(OUTPUT_SUPER_DATASET) / (1024 * 1024)
    print("\n" + "=" * 60)
    print("🏆 SUPER-DATASET CAUSAL GÉNÉRÉ")
    print("=" * 60)
    print(f"   Paires scannées       : {len(all_pairs)}")
    print(f"   Fenêtres 'Gold'       : {len(df_super)}")
    print(f"   Poids du fichier      : {size_mb:.1f} MB")
    print(f"   Chemin                : {OUTPUT_SUPER_DATASET}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Ne traiter que les N premières paires (test)")
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--slow_features", action="store_true")
    args = parser.parse_args()
    main(limit=args.limit, n_jobs=args.n_jobs, fast_features=not args.slow_features)
