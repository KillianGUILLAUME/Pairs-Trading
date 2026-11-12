"""
research/backtest/evaluate_generator.py

Batterie d'évaluation d'un générateur (Neural SDE) sur fenêtres RÉELLES hors échantillon.

Pour N fenêtres de la période de validation (split chronologique avec embargo, comme à l'entraînement),
on génère une trajectoire avec les conditions CAUSALES de la fenêtre et on compare au chemin réel :
  - volatilité du spread, volatilité et corrélation des incréments ;
  - queues (kurtosis des incréments) et clustering de volatilité (ACF des incréments²) ;
  - stationnarité (taux d'acceptation ADF) et vitesse de rappel (θ AR(1) du spread) ;
  - MMD de signature (avec canal temps) : générateur vs réel, avec deux références :
      plancher de bruit : réel vs réel (deux moitiés disjointes) ; baseline naïve : marche aléatoire gaussienne
      corrélée sans rappel à la moyenne, volatilités et corrélation calées sur le réel ;
  - utilité du conditionnement : même évaluation avec conditions permutées entre fenêtres.

    python research/backtest/evaluate_generator.py --checkpoint data/models/<run>_world_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from neural_nets.losses import KernelSignatureMMDLoss                  # noqa: E402
from neural_nets.neural_sde import GeneratorSDE                         # noqa: E402
from neural_nets.screened_loader import ScreenedPairsDataset            # noqa: E402
from neural_nets.train import chronological_split                       # noqa: E402


def adf_p(x):
    from statsmodels.tsa.stattools import adfuller
    try:
        return adfuller(x, maxlag=1, regression="c", autolag=None)[1]
    except Exception:
        return 1.0


def window_metrics(paths: np.ndarray) -> dict:
    """paths : (N, L, 2) en unités standardisées. Retourne des distributions par fenêtre."""
    A, B = paths[:, :, 0], paths[:, :, 1]
    S = A - B
    dA, dB = np.diff(A, axis=1), np.diff(B, axis=1)
    dS = np.diff(S, axis=1)
    out = {}
    out["spread_std"] = S.std(axis=1)
    out["incr_std_A"] = dA.std(axis=1)
    out["incr_std_B"] = dB.std(axis=1)
    out["incr_corr_AB"] = np.array([np.corrcoef(dA[i], dB[i])[0, 1] for i in range(len(dA))])
    out["kurt_dA"] = stats.kurtosis(dA, axis=1)
    sq = dS ** 2
    out["acf_sq_dS"] = np.array([np.corrcoef(sq[i, 1:], sq[i, :-1])[0, 1] for i in range(len(sq))])
    th = []
    for i in range(len(S)):
        b = np.polyfit(S[i, :-1], S[i, 1:], 1)[0]
        th.append(-np.log(b) if 0 < b < 1 else 0.0)
    out["theta_ar1"] = np.array(th)
    out["adf_p"] = np.array([adf_p(S[i]) for i in range(len(S))])
    out["final_spread_dev"] = np.abs(S[:, -1] - S[:, 0])
    return out


def summarize(m: dict) -> dict:
    return {
        "spread σ (med)": np.nanmedian(m["spread_std"]),
        "σ incr A (med)": np.nanmedian(m["incr_std_A"]),
        "corr incr A,B (med)": np.nanmedian(m["incr_corr_AB"]),
        "kurtosis incr (med)": np.nanmedian(m["kurt_dA"]),
        "ACF(incr²) (med)": np.nanmedian(m["acf_sq_dS"]),
        "θ AR(1) (med)": np.nanmedian(m["theta_ar1"]),
        "ADF p<0.05 (%)": 100 * np.mean(m["adf_p"] < 0.05),
        "ADF p<0.15 (%)": 100 * np.mean(m["adf_p"] < 0.15),
        "|S_fin - S_0| (med)": np.nanmedian(m["final_spread_dev"]),
    }


@torch.no_grad()
def generate(model, y0, cond, seq_len, batch=500):
    ts = torch.linspace(0.0, float(seq_len - 1), seq_len)
    out = []
    for i in range(0, len(y0), batch):
        out.append(model(y0[i:i + batch], ts, cond[i:i + batch]))
    return torch.cat(out, 0)


def gaussian_rw_baseline(real: np.ndarray, rng) -> np.ndarray:
    """Marche aléatoire gaussienne corrélée sans rappel, volatilités et corrélation calées sur l'ensemble du réel."""
    d = np.diff(real, axis=1).reshape(-1, 2)
    cov = np.cov(d.T)
    L = np.linalg.cholesky(cov)
    n, T = real.shape[0], real.shape[1] - 1
    inc = rng.standard_normal((n, T, 2)) @ L.T
    paths = np.zeros_like(real)
    paths[:, 1:, :] = np.cumsum(inc, axis=1)
    return paths


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "optimization_results", "generator_evaluation.csv"))
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, ck = GeneratorSDE.load_from_checkpoint(args.checkpoint, device="cpu")
    cfg = ck.get("config", {})
    sigcfg = ck.get("signature", {"depth": 3, "time_augment": True, "lead_lag": False})
    print(f"Checkpoint {os.path.basename(args.checkpoint)} | epoch meilleur: {ck.get('epoch', 'n/a')} | dt={model.dt} | hidden={model.drift_net[0].in_features if hasattr(model.drift_net[0], 'in_features') else '?'}")

    data_cfg = cfg.get("data", {})
    ds = ScreenedPairsDataset(args.dataset, max_adf_pvalue=float(data_cfg.get("max_adf_pvalue", 0.15)), filter_on=data_cfg.get("filter_on", "target"),
                              sig_depth=int(sigcfg["depth"]), sig_time_augment=bool(sigcfg["time_augment"]), sig_lead_lag=bool(sigcfg["lead_lag"]), device="cpu")
    train_idx, val_idx = chronological_split(ds, float(data_cfg.get("train_frac", 0.8)), int(data_cfg.get("embargo_bars", ds.seq_len)))
    print(f"{len(train_idx)} fenêtres d'entraînement, {len(val_idx)} de validation (hors échantillon, fin de période)")

    mmd_fn = KernelSignatureMMDLoss(depth=int(sigcfg["depth"]), sig_scaler_mean=ds.sig_mean, sig_scaler_std=ds.sig_std,
                                    time_augment=bool(sigcfg["time_augment"]), lead_lag=bool(sigcfg.get("lead_lag", False)))

    def mmd(real_t: torch.Tensor, gen_t: torch.Tensor) -> float:
        return float(mmd_fn(real_t, gen_t))

    results, mmds = {}, {}
    for split_name, idx_all in (("validation", val_idx), ("entraînement", train_idx)):
        idx = rng.choice(idx_all, size=min(args.n, len(idx_all)), replace=False)
        real = torch.from_numpy(ds.scaled_features[idx])
        cond = ds.metrics[idx]
        gen = generate(model, real[:, 0, :], cond, ds.seq_len)
        gen_shuf = generate(model, real[:, 0, :], cond[torch.from_numpy(rng.permutation(len(idx)))], ds.seq_len)
        base = torch.from_numpy(gaussian_rw_baseline(real.numpy(), rng)).float()
        half = len(idx) // 2
        results[(split_name, "réel")] = summarize(window_metrics(real.numpy()))
        results[(split_name, "SDE")] = summarize(window_metrics(gen.numpy()))
        results[(split_name, "SDE conditions permutées")] = summarize(window_metrics(gen_shuf.numpy()))
        results[(split_name, "marche aléatoire gaussienne")] = summarize(window_metrics(base.numpy()))
        mmds[split_name] = {
            "plancher (réel vs réel)": mmd(real[:half], real[half:2 * half]),
            "SDE": mmd(real, gen),
            "SDE conditions permutées": mmd(real, gen_shuf),
            "marche aléatoire gaussienne": mmd(real, base),
        }
        bad = int((~torch.isfinite(gen).all(dim=(1, 2))).sum())
        print(f"[{split_name}] trajectoires non finies: {bad}/{len(idx)} | |x| max généré {float(gen.abs().max()):.1f} contre réel {float(real.abs().max()):.1f}")

    df = pd.DataFrame(results).T
    pd.set_option("display.width", 220)
    for split_name in ("validation", "entraînement"):
        print(f"\n=== {split_name} : statistiques par fenêtre (médianes sauf mention) ===")
        print(df.loc[split_name].to_string(float_format=lambda v: f"{v:.3f}"))
        print(f"MMD de signature (plus petit = plus proche du réel) : " + " | ".join(f"{k}: {v:.4f}" for k, v in mmds[split_name].items()))
    out = args.out
    df.to_csv(out)
    pd.DataFrame(mmds).to_csv(out.replace(".csv", "_mmd.csv"))


if __name__ == "__main__":
    main()
