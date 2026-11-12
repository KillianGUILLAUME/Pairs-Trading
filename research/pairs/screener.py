"""
research/pairs/screener.py

Sliding-Window Screener CAUSAL.

Pour chaque fenêtre cible [s, s + window_size) on stocke :
    - les log-prix de la fenêtre cible (ce que la Neural SDE devra générer) ;
    - `cond_*`   : les 13 descripteurs calculés sur la fenêtre PASSÉE [s - past_window, s)
                   (features de conditionnement, disponibles au temps s) ;
    - `target_*` : diagnostics de la fenêtre cible (ADF, Hurst, demi-vie, OU) pour la
                   curation du dataset uniquement.

Avant novembre 2026 les descripteurs étaient calculés sur la fenêtre cible elle-même
(fuite de la cible dans le conditionnement) : les anciens fichiers `screened_*_w128_s8`
ne doivent plus servir à l'entraînement.

Usage:
    python research/pairs/screener.py                       # ADA/AVAX 1h, w=128, p=128, s=8
    python research/pairs/screener.py --pair_a ZEC_USDT --pair_b XRP_USDT --window 256 --past 256
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

from features.features_eng import get_screener_feature_dict, get_window_diagnostics


# ============================================================
# Pipeline : Analyse d'une fenêtre
# ============================================================

def analyze_window(
    idx: int,
    log_a_past: np.ndarray,
    log_b_past: np.ndarray,
    log_btc_past: np.ndarray,
    log_eth_past: np.ndarray,
    log_a_target: np.ndarray,
    log_b_target: np.ndarray,
    timestamp_cond_start: int,
    timestamp_start: int,
    timestamp_end: int,
    fast_features: bool = True,
) -> dict:
    cond = get_screener_feature_dict(log_a_past, log_b_past, log_btc_past, log_eth_past, fast=fast_features)
    diag = get_window_diagnostics(log_a_target, log_b_target)

    result = {
        "window_id": idx,
        "timestamp_cond_start": timestamp_cond_start,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "log_prices_a": log_a_target.tolist(),
        "log_prices_b": log_b_target.tolist(),
    }
    result.update({f"cond_{k}": v for k, v in cond.items()})
    result.update({f"target_{k}": v for k, v in diag.items()})
    return result


# ============================================================
# Script Principal
# ============================================================

def run_screener(
    pair_a: str = "ADA_USDT",
    pair_b: str = "AVAX_USDT",
    timeframe: str = "1h",
    window_size: int = 128,
    past_window: int = 128,
    stride: int = 8,
    n_jobs: int = -1,
    data_dir: str = "data/storage/parquet",
    output_dir: str = "data/storage/screened",
    fast_features: bool = True,
    verbose: int = 5,
):
    print(f"{'='*60}")
    print(f"🔬 SCREENER CAUSAL : {pair_a} × {pair_b} ({timeframe})")
    print(f"   Cible={window_size} barres | Passé={past_window} barres | Stride={stride} | Jobs={n_jobs}")
    print(f"{'='*60}")

    # 1. Chargement et alignement des prix
    path_a = os.path.join(data_dir, timeframe, f"{pair_a}.parquet")
    path_b = os.path.join(data_dir, timeframe, f"{pair_b}.parquet")
    path_btc = os.path.join(data_dir, timeframe, "BTC_USDT.parquet")
    path_eth = os.path.join(data_dir, timeframe, "ETH_USDT.parquet")

    for p in (path_a, path_b, path_btc, path_eth):
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ {p} introuvable. Lance le pipeline de téléchargement d'abord.")

    df_a = pd.read_parquet(path_a)[["timestamp", "close"]].rename(columns={"close": "close_a"})
    df_b = pd.read_parquet(path_b)[["timestamp", "close"]].rename(columns={"close": "close_b"})
    df_btc = pd.read_parquet(path_btc)[["timestamp", "close"]].rename(columns={"close": "close_btc"})
    df_eth = pd.read_parquet(path_eth)[["timestamp", "close"]].rename(columns={"close": "close_eth"})

    df = pd.merge(df_a, df_b, on="timestamp", how="inner")
    df = pd.merge(df, df_btc, on="timestamp", how="inner")
    df = pd.merge(df, df_eth, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    log_a_full = np.log(df["close_a"].to_numpy(dtype=float))
    log_b_full = np.log(df["close_b"].to_numpy(dtype=float))
    log_btc_full = np.log(df["close_btc"].to_numpy(dtype=float))
    log_eth_full = np.log(df["close_eth"].to_numpy(dtype=float))
    timestamps = df["timestamp"].to_numpy()

    n_total = len(log_a_full)
    print(f"📊 {n_total} barres alignées chargées.")

    # 1.5. Reprise incrémentale
    os.makedirs(output_dir, exist_ok=True)
    pair_label = f"{pair_a}x{pair_b}"
    output_path = os.path.join(output_dir, f"screened_{pair_label}_{timeframe}_w{window_size}_p{past_window}_s{stride}.parquet")

    max_existing_timestamp = 0
    df_existing = None
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_parquet(output_path)
            max_existing_timestamp = int(df_existing["timestamp_end"].max())
            print(f"🔄 Fichier existant détecté. Reprise incrémentale à partir de ts={max_existing_timestamp}")
        except Exception as e:
            print(f"⚠️ Impossible de lire {output_path} ({e}). Recalcul de zéro.")
            df_existing = None

    # 2. Découpage en fenêtres stridées : la cible commence à `start`, le passé à `start - past_window`
    all_starts = list(range(past_window, n_total - window_size + 1, stride))
    window_starts = [s for s in all_starts if int(timestamps[s + window_size - 1]) > max_existing_timestamp]

    n_windows = len(window_starts)
    if n_windows == 0:
        print("✅ Aucune nouvelle fenêtre à calculer. La paire est à jour.")
        return df_existing

    print(f"🪟 {n_windows} NOUVELLES fenêtres sur {len(all_starts)} (cible {window_size}, passé {past_window}, stride {stride})")

    # 3. Calcul parallèle
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(analyze_window)(
            idx=i,
            log_a_past=log_a_full[start - past_window:start],
            log_b_past=log_b_full[start - past_window:start],
            log_btc_past=log_btc_full[start - past_window:start],
            log_eth_past=log_eth_full[start - past_window:start],
            log_a_target=log_a_full[start:start + window_size],
            log_b_target=log_b_full[start:start + window_size],
            timestamp_cond_start=int(timestamps[start - past_window]),
            timestamp_start=int(timestamps[start]),
            timestamp_end=int(timestamps[start + window_size - 1]),
            fast_features=fast_features,
        )
        for i, start in enumerate(window_starts)
    )
    elapsed = time.time() - t0
    print(f"⏱️  Calcul terminé en {elapsed:.1f}s ({n_windows / max(elapsed, 1e-9):.0f} fenêtres/sec)")

    # 4. DataFrame
    new_df = pd.DataFrame(results)
    if df_existing is not None and not new_df.empty:
        df_screened = pd.concat([df_existing, new_df], ignore_index=True)
    else:
        df_screened = new_df

    # 5. Statistiques
    n_stat = int((df_screened["target_adf_pvalue"] < 0.05).sum())
    n_mr = int((df_screened["target_hurst"] < 0.5).sum())
    hl = df_screened["target_half_life"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n{'='*60}")
    print("📈 RÉSULTATS DU SCREENING (fenêtres cibles)")
    print(f"{'='*60}")
    print(f"   Stationnaires (ADF p<0.05) : {n_stat}/{len(df_screened)} ({100*n_stat/len(df_screened):.1f}%)")
    print(f"   Mean-reverting (H<0.5)     : {n_mr}/{len(df_screened)} ({100*n_mr/len(df_screened):.1f}%)")
    print(f"   Demi-vie médiane            : {hl.median():.1f} barres")
    print(f"   Hurst moyen (cible)         : {df_screened['target_hurst'].mean():.3f} | Hurst moyen (passé) : {df_screened['cond_hurst'].mean():.3f}")

    # 6. Sauvegarde
    df_screened.to_parquet(output_path, engine="pyarrow", index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n💾 Sauvegardé : {output_path} ({size_mb:.1f} MB)")

    return df_screened


# ============================================================
# Multi-Window
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
    if window_sizes is None:
        window_sizes = [64, 128, 256, 512]

    summary_rows = []
    for w in window_sizes:
        print(f"\n{'='*60}\n🔬 FENÊTRE = {w} barres\n{'='*60}")
        df = run_screener(pair_a=pair_a, pair_b=pair_b, timeframe=timeframe, window_size=w, past_window=w,
                          stride=stride, n_jobs=n_jobs, data_dir=data_dir, output_dir=output_dir)
        gold = df[df["target_adf_pvalue"] < 0.05]
        hl_gold = gold["target_half_life"].replace([np.inf, -np.inf], np.nan).dropna()
        summary_rows.append({
            "window": w,
            "n_total": len(df),
            "n_stationary": len(gold),
            "pct_stationary": round(100 * len(gold) / len(df), 1) if len(df) > 0 else 0,
            "hurst_all_mean": round(df["target_hurst"].mean(), 3),
            "hurst_gold_mean": round(gold["target_hurst"].mean(), 3) if len(gold) > 0 else None,
            "half_life_gold_median": round(hl_gold.median(), 1) if len(hl_gold) > 0 else None,
            "theta_gold_mean": round(gold["target_ou_theta"].mean(), 4) if len(gold) > 0 else None,
            "sigma_gold_mean": round(gold["target_ou_sigma"].mean(), 4) if len(gold) > 0 else None,
        })

    summary = pd.DataFrame(summary_rows)
    print(f"\n\n{'='*80}\n📊 COMPARAISON MULTI-FENÊTRES — {pair_a} × {pair_b}\n{'='*80}")
    print(summary.to_string(index=False))
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"multi_window_{pair_a}x{pair_b}_{timeframe}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n💾 Résumé : {summary_path}")
    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screener causal de fenêtres cointégrées")
    parser.add_argument("--pair_a", type=str, default="ADA_USDT")
    parser.add_argument("--pair_b", type=str, default="AVAX_USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--past", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--slow_features", action="store_true", help="Estimateurs MLE (lents) pour ν et skew")
    parser.add_argument("--multi", action="store_true", help="Lance sur plusieurs tailles (64,128,256,512)")
    parser.add_argument("--windows", type=str, default="64,128,256,512")
    args = parser.parse_args()

    data_dir = os.path.join(project_root, "data", "storage", "parquet")
    output_dir = os.path.join(project_root, "data", "storage", "screened")

    if args.multi:
        sizes = [int(x) for x in args.windows.split(",")]
        run_multi_window(pair_a=args.pair_a, pair_b=args.pair_b, timeframe=args.timeframe, window_sizes=sizes,
                         stride=args.stride, n_jobs=args.n_jobs, data_dir=data_dir, output_dir=output_dir)
    else:
        run_screener(pair_a=args.pair_a, pair_b=args.pair_b, timeframe=args.timeframe,
                     window_size=args.window, past_window=args.past, stride=args.stride,
                     n_jobs=args.n_jobs, data_dir=data_dir, output_dir=output_dir,
                     fast_features=not args.slow_features)
