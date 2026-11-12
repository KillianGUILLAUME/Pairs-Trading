"""
research/backtest/real_reversion_check.py

La calibration OU prédit-elle le retour à la moyenne RÉEL hors échantillon ?
Pour chaque barre t où le moteur est calibré, la prédiction OU de la variation du spread sur h barres est
    pred_h = -(1 - exp(-θ̂ h)) (x_t - μ̂_t)
(θ̂, μ̂ : dernière calibration, passé uniquement). On régresse la variation réalisée x_{t+h} - x_t sur pred_h :
pente 1 = l'OU calibré prédit exactement ; 0 = aucun retour ; < 0 = tendance.
On mesure aussi la dérive du niveau moyen : (moyenne des 200 barres suivantes) - μ̂_t, en écarts-types σ̂∞.
Les barres sont restreintes aux situations où un signal existe (|x_t - μ̂_t| > 0,5 σ̂∞).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.backtest.real_walk_forward import load_pair                      # noqa: E402
from research.institutional.signal_engine import InstitutionalSignalEngine    # noqa: E402

HORIZONS = (1, 6, 24, 72)


def run_pair(a: str, b: str):
    from loguru import logger
    logger.remove()
    df = load_pair(a, b)
    if len(df) < 2000:
        return None
    ts = df["timestamp"].to_numpy()
    sig = InstitutionalSignalEngine(solver_type="ergodic", cost=0.002).generate(ts, df["a"].to_numpy(float), df["b"].to_numpy(float))
    x, mu, th, sg = sig.spreads, sig.mu_ref, sig.theta_hat, sig.sigma_hat
    sinf = sg / np.sqrt(2 * th)
    z = (x - mu) / sinf
    n = len(x)
    year = pd.to_datetime(ts, unit="ms").year.to_numpy()
    out = {"pair": f"{a}x{b}"}
    for h in HORIZONS:
        idx = np.arange(n - h)
        pred = -(1 - np.exp(-th[idx] * h)) * (x[idx] - mu[idx])
        real = x[idx + h] - x[idx]
        ok = np.isfinite(pred) & (np.abs(z[idx]) > 0.5)
        out[f"num_{h}"] = float(np.sum(pred[ok] * real[ok]))
        out[f"den_{h}"] = float(np.sum(pred[ok] ** 2))
        out[f"n_{h}"] = int(ok.sum())
        for y in (2022, 2023):
            oky = ok & (year[idx] == y)
            out[f"num_{h}_{y}"] = float(np.sum(pred[oky] * real[oky]))
            out[f"den_{h}_{y}"] = float(np.sum(pred[oky] ** 2))
    # dérive du niveau moyen sur 200 barres
    H = 200
    idx = np.arange(n - H)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    fut_mean = (csum[idx + H + 1] - csum[idx + 1]) / H
    drift = (fut_mean - mu[idx]) / sinf[idx]
    d = drift[np.isfinite(drift)]
    out["drift_abs_med"] = float(np.median(np.abs(d))) if len(d) else np.nan
    out["drift_abs_p90"] = float(np.percentile(np.abs(d), 90)) if len(d) else np.nan
    return out


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    args = ap.parse_args()
    pairs = pd.read_parquet(args.dataset, columns=["pair_a", "pair_b"]).drop_duplicates().values.tolist()
    if args.limit:
        pairs = pairs[:args.limit]
    rows = [r for r in Parallel(n_jobs=args.jobs, verbose=1)(delayed(run_pair)(a, b) for a, b in pairs) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "optimization_results", "real_reversion_check.csv"), index=False)
    print(f"\n{len(df)} paires. Pente = réalisé / prédit par l'OU calibré (1 = parfait, 0 = pas de retour, <0 = tendance)")
    print(f"{'horizon':>8}{'pente poolée':>14}{'2022':>8}{'2023':>8}{'médiane des paires':>20}{'paires pente>0':>16}")
    for h in HORIZONS:
        pooled = df[f"num_{h}"].sum() / df[f"den_{h}"].sum()
        s22 = df[f"num_{h}_2022"].sum() / df[f"den_{h}_2022"].sum()
        s23 = df[f"num_{h}_2023"].sum() / df[f"den_{h}_2023"].sum()
        per = (df[f"num_{h}"] / df[f"den_{h}"])
        print(f"{h:>7}h{pooled:>14.3f}{s22:>8.3f}{s23:>8.3f}{per.median():>20.3f}{(per > 0).mean():>16.0%}")
    print(f"\ndérive du niveau moyen sur 200 barres, en σ∞ : médiane |dérive| {df.drift_abs_med.median():.2f} | 90e centile {df.drift_abs_p90.median():.2f}  (largeur de bande optimale ~0,5 σ∞)")


if __name__ == "__main__":
    main()
