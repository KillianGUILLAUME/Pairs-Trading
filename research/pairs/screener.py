"""
neural_nets/screener.py

Sliding-Window Screener : 
Extrait les métriques de cointégration (ADF, Hurst, Half-Life) 
sur des fenêtres glissantes et stocke le tout dans un fichier Parquet.

Usage:
    python -m neural_nets.screener                     # Défaut ADA/AVAX 1h
    python -m neural_nets.screener --pair_a ZEC_USDT --pair_b XRP_USDT --window 256
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from features.features_eng import get_screener_feature_dict


# ============================================================
# Pipeline : Analyse d'une fenêtre
# ============================================================

def analyze_window(
    idx: int,
    log_a: np.ndarray, 
    log_b: np.ndarray,
    log_btc: np.ndarray,
    log_eth: np.ndarray,
    timestamp_start: int,
    timestamp_end: int,
) -> dict:
    """
    Calcule les métriques de cointégration via le module features_eng.
    Retourne un dict complet.
    """
    feature_dict = get_screener_feature_dict(log_a, log_b, log_btc, log_eth)
    
    result = {
        "window_id": idx,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "log_prices_a": log_a.tolist(),
        "log_prices_b": log_b.tolist(),
    }
    result.update(feature_dict)
    
    return result


# ============================================================
# Script Principal
# ============================================================

def run_screener(
    pair_a: str = "ADA_USDT",
    pair_b: str = "AVAX_USDT",
    timeframe: str = "1h",
    window_size: int = 128,
    stride: int = 8,
    n_jobs: int = -1,
    data_dir: str = "data/storage/parquet",
    output_dir: str = "data/storage/screened",
):
    """
    Pipeline principal du screener.
    """
    print(f"{'='*60}")
    print(f"🔬 SCREENER : {pair_a} × {pair_b} ({timeframe})")
    print(f"   Window={window_size} bars | Stride={stride} | Jobs={n_jobs}")
    print(f"{'='*60}")
    
    # 1. Chargement et alignement des prix
    path_a = os.path.join(data_dir, timeframe, f"{pair_a}.parquet")
    path_b = os.path.join(data_dir, timeframe, f"{pair_b}.parquet")
    path_btc = os.path.join(data_dir, timeframe, "BTC_USDT.parquet")
    path_eth = os.path.join(data_dir, timeframe, "ETH_USDT.parquet")
    
    if not os.path.exists(path_a) or not os.path.exists(path_btc) or not os.path.exists(path_eth):
        raise FileNotFoundError(f"❌ {path_a} ou BTC_USDT ou ETH_USDT introuvables. Lance le pipeline de téléchargement d'abord.")
    if not os.path.exists(path_b):
        raise FileNotFoundError(f"❌ {path_b} introuvable.")
    
    df_a = pd.read_parquet(path_a)[["timestamp", "close"]].rename(columns={"close": "close_a"})
    df_b = pd.read_parquet(path_b)[["timestamp", "close"]].rename(columns={"close": "close_b"})
    df_btc = pd.read_parquet(path_btc)[["timestamp", "close"]].rename(columns={"close": "close_btc"})
    df_eth = pd.read_parquet(path_eth)[["timestamp", "close"]].rename(columns={"close": "close_eth"})
    
    df = pd.merge(df_a, df_b, on="timestamp", how="inner")
    df = pd.merge(df, df_btc, on="timestamp", how="inner")
    df = pd.merge(df, df_eth, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    price_a = df["close_a"].to_numpy()
    price_b = df["close_b"].to_numpy()
    price_btc = df["close_btc"].to_numpy()
    price_eth = df["close_eth"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    
    log_a_full = np.log(price_a)
    log_b_full = np.log(price_b)
    log_btc_full = np.log(price_btc)
    log_eth_full = np.log(price_eth)
    
    n_total = len(log_a_full)
    print(f"📊 {n_total} barres alignées chargées.")
    
    # 1.5. Vérification du fichier existant (Incremental Update)
    os.makedirs(output_dir, exist_ok=True)
    pair_label = f"{pair_a}x{pair_b}"
    output_path = os.path.join(output_dir, f"screened_{pair_label}_{timeframe}_w{window_size}_s{stride}.parquet")
    
    max_existing_timestamp = 0
    df_existing = None
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_parquet(output_path, engine="fastparquet")
            max_existing_timestamp = df_existing["timestamp_end"].max()
            print(f"🔄 Fichier existant détecté. Reprise incrémentale à partir de ts={max_existing_timestamp}")
        except Exception as e:
            print(f"⚠️ Impossible de lire {output_path} ({e}). Recalcul de zéro.")

    # 2. Découpage en fenêtres stridées
    window_starts = []
    for start in range(0, n_total - window_size + 1, stride):
        ts_end = int(timestamps[start + window_size - 1])
        if ts_end > max_existing_timestamp:
            window_starts.append(start)
            
    n_windows = len(window_starts)
    if n_windows == 0:
        print(f"✅ Aucune nouvelle fenêtre à calculer. La paire est à jour.")
        return df_existing
        
    print(f"🪟 {n_windows} NOUVELLES fenêtres sur {len(range(0, n_total - window_size + 1, stride))} de {window_size} barres (stride={stride})")
    
    # 3. Calcul parallèle des métriques
    t0 = time.time()
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(analyze_window)(
            idx=i,
            log_a=log_a_full[start : start + window_size],
            log_b=log_b_full[start : start + window_size],
            log_btc=log_btc_full[start : start + window_size],
            log_eth=log_eth_full[start : start + window_size],
            timestamp_start=int(timestamps[start]),
            timestamp_end=int(timestamps[start + window_size - 1]),
        )
        for i, start in enumerate(window_starts)
    )
    
    elapsed = time.time() - t0
    print(f"⏱️  Calcul terminé en {elapsed:.1f}s ({n_windows / elapsed:.0f} fenêtres/sec)")
    
    # 4. Construction du DataFrame
    new_df = pd.DataFrame(results)
    if df_existing is not None and not new_df.empty:
        df_screened = pd.concat([df_existing, new_df], ignore_index=True)
    elif df_existing is not None:
        df_screened = df_existing
    else:
        df_screened = new_df
    
    # 5. Statistiques rapides
    n_stationary = (df_screened["adf_pvalue"] < 0.05).sum()
    n_mean_reverting = (df_screened["hurst"] < 0.5).sum()
    median_hl = df_screened.loc[df_screened["half_life"] < float("inf"), "half_life"].median()
    
    print(f"\n{'='*60}")
    print(f"📈 RÉSULTATS DU SCREENING")
    print(f"{'='*60}")
    print(f"   Fenêtres stationnaires (ADF p<0.05) : {n_stationary}/{n_windows} ({100*n_stationary/n_windows:.1f}%)")
    print(f"   Fenêtres mean-reverting (H<0.5)     : {n_mean_reverting}/{n_windows} ({100*n_mean_reverting/n_windows:.1f}%)")
    print(f"   Half-life médiane                    : {median_hl:.1f} barres")
    print(f"   Hurst moyen                          : {df_screened['hurst'].mean():.3f}")
    print(f"   ADF p-value médiane                  : {df_screened['adf_pvalue'].median():.4f}")
    
    # 6. Sauvegarde en Parquet
    os.makedirs(output_dir, exist_ok=True)
    
    # (output_path est déjà géré en amont pour l'incremental update)
    
    df_screened.to_parquet(output_path, engine="pyarrow", index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n💾 Sauvegardé : {output_path} ({size_mb:.1f} MB)")
    
    return df_screened


# ============================================================
# Multi-Window : Comparaison de tailles de fenêtres
# ============================================================

def run_multi_window(
    pair_a: str = "ADA_USDT",
    pair_b: str = "AVAX_USDT",
    timeframe: str = "1h",
    window_sizes: list = None,
    stride: int = 8,
    n_jobs: int = -1,
    data_dir: str = "data/storage/parquet",
    output_dir: str = "data/storage/screened",
):
    """
    Lance le screener sur plusieurs tailles de fenêtres et affiche le tableau comparatif.
    """
    if window_sizes is None:
        window_sizes = [64, 128, 256, 512]
    
    summary_rows = []
    
    for w in window_sizes:
        print(f"\n{'='*60}")
        print(f"🔬 FENÊTRE = {w} barres")
        print(f"{'='*60}")
        
        df = run_screener(
            pair_a=pair_a, pair_b=pair_b, timeframe=timeframe,
            window_size=w, stride=stride, n_jobs=n_jobs,
            data_dir=data_dir, output_dir=output_dir,
        )
        
        # Statistiques sur les fenêtres stationnaires (pépites)
        gold = df[df["adf_pvalue"] < 0.05]
        
        row = {
            "window": w,
            "n_total": len(df),
            "n_stationary": len(gold),
            "pct_stationary": round(100 * len(gold) / len(df), 1) if len(df) > 0 else 0,
            "hurst_all_mean": round(df["hurst"].mean(), 3),
            "hurst_gold_mean": round(gold["hurst"].mean(), 3) if len(gold) > 0 else None,
            "half_life_gold_median": round(gold.loc[gold["half_life"] < 1e6, "half_life"].median(), 1) if len(gold) > 0 else None,
            "theta_gold_mean": round(gold["ou_theta"].mean(), 4) if len(gold) > 0 else None,
            "theta_gold_std": round(gold["ou_theta"].std(), 4) if len(gold) > 0 else None,
            "sigma_gold_mean": round(gold["ou_sigma"].mean(), 4) if len(gold) > 0 else None,
            "sigma_gold_std": round(gold["ou_sigma"].std(), 4) if len(gold) > 0 else None,
        }
        summary_rows.append(row)
    
    summary = pd.DataFrame(summary_rows)
    
    print(f"\n\n{'='*80}")
    print(f"📊 COMPARAISON MULTI-FENÊTRES — {pair_a} × {pair_b}")
    print(f"{'='*80}")
    print(summary.to_string(index=False))
    print(f"{'='*80}")
    
    # Sauvegarde du résumé
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"multi_window_{pair_a}x{pair_b}_{timeframe}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n💾 Résumé : {summary_path}")
    
    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screener de fenêtres cointégrées")
    parser.add_argument("--pair_a", type=str, default="ADA_USDT")
    parser.add_argument("--pair_b", type=str, default="AVAX_USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--multi", action="store_true", help="Lance sur plusieurs tailles (64,128,256,512)")
    parser.add_argument("--windows", type=str, default="64,128,256,512", help="Tailles (séparées par des virgules)")
    args = parser.parse_args()
    
    data_dir = os.path.join(project_root, "data", "storage", "parquet")
    output_dir = os.path.join(project_root, "data", "storage", "screened")
    
    if args.multi:
        sizes = [int(x) for x in args.windows.split(",")]
        run_multi_window(
            pair_a=args.pair_a, pair_b=args.pair_b,
            timeframe=args.timeframe, window_sizes=sizes,
            stride=args.stride, n_jobs=args.n_jobs,
            data_dir=data_dir, output_dir=output_dir,
        )
    else:
        run_screener(
            pair_a=args.pair_a, pair_b=args.pair_b,
            timeframe=args.timeframe,
            window_size=args.window, stride=args.stride,
            n_jobs=args.n_jobs, data_dir=data_dir, output_dir=output_dir,
        )
