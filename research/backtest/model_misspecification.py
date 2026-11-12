"""
research/backtest/model_misspecification.py

Combien coûte l'hypothèse « le spread est un OU » ? On simule des marchés dont le spread
s'écarte de l'OU de façon calibrée sur les données réelles (queues épaisses, volatilité
stochastique, sauts, dérive du niveau moyen, régimes de vitesse), on fait tourner le
moteur de signaux HJB tel quel, et on le compare à :

  - ORACLE : mêmes solveurs mais avec les VRAIS paramètres locaux (θ_t, σ_t, μ_t) connus
             à chaque barre. Mesure le coût de l'estimation et de la dérive des paramètres.
  - k* : échelle de bande optimale a posteriori (seuils du moteur multipliés par k, k choisi
         pour maximiser le PnL moyen sur les trajectoires). k* = 1 veut dire que les seuils
         OU sont déjà les bons ; k* ≠ 1 mesure le biais de forme de l'OU.

Usage :
    python research/backtest/model_misspecification.py --paths 40 --bars 4000
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.backtest.engine import BacktestConfig, BacktestEngine          # noqa: E402
from research.institutional.signal_engine import InstitutionalSignalEngine   # noqa: E402


# ============================================================================
# Simulateur de spread
# ============================================================================

@dataclass
class SpreadDGP:
    """Paramètres par barre. Toutes les extensions sont désactivées par défaut (= OU pur)."""
    theta: float = 0.02              # vitesse de retour (par barre) ; l'estimation AR(1) sur 156 barres la gonfle (~x1.6)
    sigma_resid: float = 0.0060      # écart-type des innovations (60 bp)
    student_nu: Optional[float] = None       # None = gaussien ; sinon innovations de Student(ν) à variance unité
    sv_vol_of_vol: float = 0.0               # >0 : volatilité stochastique, log σ_t AR(1)
    sv_persistence: float = 0.98
    jump_prob: float = 0.0
    jump_sd_mult: float = 5.0        # écart-type des sauts, en multiples de sigma_resid
    mu_step_mult: float = 0.0        # >0 : le niveau moyen μ_t suit une marche aléatoire (pas = mult × sigma_resid)
    regime_thetas: Optional[tuple] = None    # (θ_haut, θ_bas) : chaîne de Markov à 2 états
    regime_stay: float = 0.995
    break_frac: Optional[float] = None       # rupture de cointégration : après cette fraction de la série, θ -> ~0 (marche aléatoire)


DGPS: Dict[str, SpreadDGP] = {
    "OU pur": SpreadDGP(),
    "+ queues Student": SpreadDGP(student_nu=5.0),
    "+ vol stochastique": SpreadDGP(sv_vol_of_vol=0.18),
    "+ sauts": SpreadDGP(jump_prob=0.008),
    "+ niveau moyen dérivant": SpreadDGP(mu_step_mult=0.08),
    "+ régimes de θ": SpreadDGP(regime_thetas=(0.05, 0.008)),
    "rupture de cointégration": SpreadDGP(break_frac=0.6),
    "tout combiné": SpreadDGP(student_nu=8.0, sv_vol_of_vol=0.15, jump_prob=0.004, mu_step_mult=0.08,
                              regime_thetas=(0.05, 0.008)),
}


def simulate_spread(dgp: SpreadDGP, n: int, seed: int):
    """
    Retourne (x, truth) : spread simulé et vérité locale par barre
    (theta_t, mu_t, sigma_t = écart-type de la partie diffusive des innovations).
    """
    rng = np.random.default_rng(seed)
    if dgp.student_nu:
        eps = rng.standard_t(dgp.student_nu, n) / np.sqrt(dgp.student_nu / (dgp.student_nu - 2.0))
    else:
        eps = rng.standard_normal(n)
    if dgp.sv_vol_of_vol > 0:
        h = np.zeros(n)
        eta = rng.standard_normal(n)
        for t in range(1, n):
            h[t] = dgp.sv_persistence * h[t - 1] + dgp.sv_vol_of_vol * eta[t]
        var_h = dgp.sv_vol_of_vol ** 2 / (1.0 - dgp.sv_persistence ** 2)
        sig_t = dgp.sigma_resid * np.exp(h - var_h / 2.0)
    else:
        sig_t = np.full(n, dgp.sigma_resid)
    if dgp.regime_thetas:
        state = np.zeros(n, dtype=int)
        u = rng.random(n)
        for t in range(1, n):
            state[t] = state[t - 1] if u[t] < dgp.regime_stay else 1 - state[t - 1]
        theta_t = np.where(state == 0, dgp.regime_thetas[0], dgp.regime_thetas[1])
    else:
        theta_t = np.full(n, dgp.theta)
    if dgp.break_frac is not None:
        theta_t = theta_t.copy()
        theta_t[int(dgp.break_frac * n):] = 1e-4
    mu_t = np.zeros(n)
    if dgp.mu_step_mult > 0:
        mu_t = np.cumsum(rng.standard_normal(n) * dgp.mu_step_mult * dgp.sigma_resid)
    jumps = (rng.random(n) < dgp.jump_prob) * rng.standard_normal(n) * dgp.jump_sd_mult * dgp.sigma_resid

    x = np.zeros(n)
    for t in range(1, n):
        b = np.exp(-theta_t[t])
        x[t] = mu_t[t] + b * (x[t - 1] - mu_t[t - 1]) + sig_t[t] * eps[t] + jumps[t]
    return x, {"theta": theta_t, "mu": mu_t, "sigma_resid": sig_t}


def spread_to_prices(x: np.ndarray, seed: int, const: float = 0.5, beta: float = 1.0):
    """log A = beta * log B + const + x, log B marche aléatoire (1 %/barre)."""
    rng = np.random.default_rng(10_000 + seed)
    log_b = np.cumsum(0.01 * rng.standard_normal(len(x))) + 3.0
    return np.exp(beta * log_b + const + x), np.exp(log_b)


def ou_sigma_from_resid(theta: float, sigma_resid: float) -> float:
    """σ continu équivalent d'un AR(1) de pas 1 : σ = σ_resid · sqrt(-2 ln b / (1 - b²))."""
    b = np.exp(-theta)
    return sigma_resid * np.sqrt(2.0 * theta / (1.0 - b ** 2))


def describe(x: np.ndarray, window: int = 156) -> Dict[str, float]:
    """Mêmes indicateurs que ceux mesurés sur le spread réel ADA/AVAX."""
    from scipy import stats
    th, kurt, arch = [], [], []
    for i in range(0, len(x) - window - 1, 24):
        xs = x[i:i + window + 1]
        bb, aa = np.polyfit(xs[:-1], xs[1:], 1)
        r = xs[1:] - (aa + bb * xs[:-1])
        if 0 < bb < 1:
            th.append(-np.log(bb))
        kurt.append(stats.kurtosis(r))
        arch.append(np.corrcoef(r[1:] ** 2, r[:-1] ** 2)[0, 1])
    bb, aa = np.polyfit(x[:-1], x[1:], 1)
    return {
        "theta_med": float(np.median(th)),
        "theta_p90_p10": float(np.percentile(th, 90) / np.percentile(th, 10)),
        "kurt_med": float(np.median(kurt)),
        "acf_sq_med": float(np.nanmedian(arch)),
        "theta_full": float(-np.log(bb)) if 0 < bb < 1 else 0.0,
    }


# ============================================================================
# Politiques
# ============================================================================

def rescale_signal(sig, k: float):
    """Multiplie par k les écarts des seuils à μ (entrées ET sorties) et recalcule les signaux."""
    mus = sig.mu_ref if sig.mu_ref is not None else sig.mus
    L_in = mus + k * (sig.d_star - mus)
    S_in = mus + k * (sig.b_star - mus)
    L_out = mus + k * (sig.long_exit_level - mus)
    S_out = mus + k * (sig.short_exit_level - mus)
    s = sig.spreads
    with np.errstate(invalid="ignore"):
        return replace(
            sig,
            entry_long=(s < L_in).astype(float), entry_short=(s > S_in).astype(float),
            exit_long=(s >= L_out).astype(float), exit_short=(s <= S_out).astype(float),
        )


def pnl_of(sig, start: int, cost_cfg: BacktestConfig) -> Dict[str, float]:
    res = BacktestEngine(cost_cfg).run(sig.slice(start, len(sig.spreads)))
    m = res["metrics"]
    return {"ret": m["total_return_pct"], "sharpe": m["sharpe"], "trades": m["n_trades"], "in_mkt": m["pct_in_market"]}


K_GRID = (0.7, 1.0, 1.5)
BETA_TRUE = 1.5
CONST = 0.5


def run_one(dgp_name: str, seed: int, bars: int, warmup: int, cost: float):
    """
    Trois configurations sur la même trajectoire (β vrai = 1,5, constante 0,5) :
      legacy : couverture Kalman (innovation), seuils sur EMA rapide, fenêtre 156, arrêt temporel 2 x HL
      new    : couverture OLS roulante, μ calibré, fenêtre 624, arrêt temporel 8 x HL (défauts actuels)
      oracle : mêmes réglages que new mais β et (θ_t, μ_t, σ_t) VRAIS
    et, pour `new`, l'échelle k des bandes (k = 1 : seuils de la théorie).
    """
    dgp = DGPS[dgp_name]
    x, truth = simulate_spread(dgp, bars, seed)
    pa, pb = spread_to_prices(x, seed, const=CONST, beta=BETA_TRUE)
    ts = np.arange(bars)

    legacy = InstitutionalSignalEngine(window=156, solver_type="ergodic", cost=cost, use_kalman=True,
                                       mean_source="ema", hedge="kalman").generate(ts, pa, pb)
    new = InstitutionalSignalEngine(solver_type="ergodic", cost=cost).generate(ts, pa, pb)

    def oracle(i):
        th = float(truth["theta"][i])
        return th, CONST + float(truth["mu"][i]), float(ou_sigma_from_resid(th, truth["sigma_resid"][i]))
    orc = InstitutionalSignalEngine(solver_type="ergodic", cost=cost, fixed_beta=BETA_TRUE, oracle_params=oracle).generate(ts, pa, pb)

    cfg_legacy = BacktestConfig(position_size=1.0, time_stop_mult=2.0)
    cfg_new = BacktestConfig(position_size=1.0)
    out = {"dgp": dgp_name, "seed": seed,
           "legacy": pnl_of(legacy, warmup, cfg_legacy)["ret"],
           "oracle": pnl_of(orc, warmup, cfg_new)["ret"]}
    for k in K_GRID:
        out[f"k={k}"] = pnl_of(rescale_signal(new, k), warmup, cfg_new)["ret"]
    out["new"] = out["k=1.0"]
    return out


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--bars", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=1100)
    ap.add_argument("--cost", type=float, default=0.002)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "optimization_results", "misspecification.csv"))
    args = ap.parse_args()

    jobs = [(name, s) for name in DGPS for s in range(args.paths)]
    rows = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(run_one)(name, s, args.bars, args.warmup, args.cost) for s in range(args.paths) for name in DGPS)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{args.paths} trajectoires x {args.bars} barres par scénario ; rendement total moyen en % (position = 100 % du capital, beta vrai {BETA_TRUE})")
    print(f"{'scénario':<26}{'historique':>11}{'nouveau':>9}{'k=0.7':>8}{'k=1.5':>8}{'oracle':>9}{'nouv./oracle':>14}{'gagnants (nouv.)':>18}")
    for name in DGPS:
        d = df[df.dgp == name]
        print(f"{name:<26}{d.legacy.mean():>11.1f}{d.new.mean():>9.1f}{d['k=0.7'].mean():>8.1f}{d['k=1.5'].mean():>8.1f}"
              f"{d.oracle.mean():>9.1f}{d.new.mean() / d.oracle.mean():>14.0%}{(d.new > 0).mean():>18.0%}")


if __name__ == "__main__":
    main()
