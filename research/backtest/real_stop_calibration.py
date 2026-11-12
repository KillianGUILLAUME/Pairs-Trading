"""
research/backtest/real_stop_calibration.py

Calibration du stop adverse sur paires réelles, avec séparation chronologique stricte :
  - 2022 : calibration (excursions adverses des trades, puis balayage du rendement) ;
  - 2023 : évaluation, jamais regardée pour choisir.
Critère fixé AVANT les résultats : s* = argmax du Sharpe net du portefeuille équipondéré sur 2022 ;
s* n'est « validé » que s'il améliore aussi le Sharpe net 2023 par rapport à l'absence de stop.

Le stop s'exprime en écarts-types stationnaires sigma_inf au-delà de la bande d'entrée, sur le spread à
couverture verrouillée (voir locked_policy.py). Couverture verrouillée, arrêt temporel à 8 demi-vies conservé.
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

from research.backtest.engine import BacktestConfig, BacktestEngine          # noqa: E402
from research.backtest.real_walk_forward import load_pair, bar_returns, summarize  # noqa: E402
from research.institutional.signal_engine import InstitutionalSignalEngine   # noqa: E402

STOP_GRID = (None, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
NET = dict(fee_rate=0.0004, slippage_bps=1.0)


def run_pair(a: str, b: str):
    from loguru import logger
    logger.remove()
    df = load_pair(a, b)
    if len(df) < 2000:
        return None
    ts = df["timestamp"].to_numpy()
    sig = InstitutionalSignalEngine(solver_type="ergodic", cost=0.002).generate(ts, df["a"].to_numpy(float), df["b"].to_numpy(float))
    cfg = BacktestConfig(position_size=0.10, time_stop_mult=8.0, **NET)
    series, stats = {}, {}
    trades_base = None
    for s in STOP_GRID:
        res = BacktestEngine(cfg).run(sig, lock_hedge=True, stop_sd=s)
        series[s] = bar_returns(res["df"]).astype(np.float32)
        stats[s] = {"trades": res["metrics"]["n_trades"], "ret": res["metrics"]["total_return_pct"]}
        if s is None:
            bt = res["df"]
            spr = bt["ret_a"].to_numpy() - bt["beta_r"].to_numpy() * bt["ret_b"].to_numpy()
            rows = []
            for t in res["trades"]:
                e, x = t["entry"], t["exit"]
                rows.append({"pair": f"{a}x{b}", "year": int(pd.to_datetime(ts[e], unit="ms").year), "kind": t["kind"],
                             "mae_sd": t["mae_sd"], "pnl": float(t["side"] * np.sum(spr[e + 1:x + 1]))})
            trades_base = rows
    return f"{a}x{b}", ts, series, stats, trades_base


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

    # ---- 1) excursions adverses des trades sans stop, 2022 seulement (calibration) ----
    tr = pd.DataFrame([t for r in rows for t in r[4]])
    tr.to_csv(os.path.join(PROJECT_ROOT, "data", "optimization_results", "stop_calibration_trades.csv"), index=False)
    cal = tr[tr.year == 2022]
    win, lose = cal[cal.pnl > 0], cal[cal.pnl <= 0]
    print(f"\n=== Calibration 2022 : {len(cal)} trades sans stop ({len(win)} gagnants, {len(lose)} perdants) ===")
    qs = (0.5, 0.75, 0.9, 0.95, 0.99)
    print("excursion adverse maximale (en sigma_inf au-delà de la bande d'entrée) :")
    print("  gagnants  : " + " | ".join(f"q{int(q*100)}={win.mae_sd.quantile(q):.2f}" for q in qs))
    print("  perdants  : " + " | ".join(f"q{int(q*100)}={lose.mae_sd.quantile(q):.2f}" for q in qs))
    print("pour chaque stop s : part des gagnants coupés à tort / part des perdants coupés / gain moyen des perdants coupés évité (en pb)")
    for s in STOP_GRID[1:]:
        cut_w = float((win.mae_sd >= s).mean()); cut_l = float((lose.mae_sd >= s).mean())
        print(f"  s={s:>3}: gagnants coupés {cut_w:5.1%} | perdants coupés {cut_l:5.1%}")

    # ---- 2) balayage du rendement, 2022 (choix) puis 2023 (évaluation) ----
    out = []
    for s in STOP_GRID:
        rets = pd.DataFrame({r[0]: pd.Series(r[2][s], index=r[1]) for r in rows}).sort_index()
        year = pd.to_datetime(rets.index, unit="ms").year
        a22, a23 = summarize(rets[year == 2022], ""), summarize(rets[year == 2023], "")
        out.append({"stop_sd": "aucun" if s is None else s,
                    "2022 total_%": a22["total_%"], "2022 sharpe": a22["sharpe"], "2022 max_dd_%": a22["max_dd_%"],
                    "2023 total_%": a23["total_%"], "2023 sharpe": a23["sharpe"], "2023 max_dd_%": a23["max_dd_%"],
                    "trades médian": float(np.median([r[3][s]["trades"] for r in rows])),
                    "paires gagnantes %": 100 * float(np.mean([r[3][s]["ret"] > 0 for r in rows]))})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(PROJECT_ROOT, "data", "optimization_results", "stop_calibration.csv"), index=False)
    print(f"\n=== Portefeuille équipondéré ({len(rows)} paires), net de coûts ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    cand = res[res.stop_sd != "aucun"].copy()
    best = cand.loc[cand["2022 sharpe"].idxmax()]
    base = res[res.stop_sd == "aucun"].iloc[0]
    print(f"\ns* choisi sur 2022 (Sharpe net maximal) = {best.stop_sd}")
    print(f"  2022 : Sharpe {best['2022 sharpe']:.2f} (sans stop {base['2022 sharpe']:.2f})")
    print(f"  2023 : Sharpe {best['2023 sharpe']:.2f} (sans stop {base['2023 sharpe']:.2f}) ; rendement {best['2023 total_%']:.2f}% (sans stop {base['2023 total_%']:.2f}%)")
    print("  validé" if best["2023 sharpe"] > base["2023 sharpe"] and best["2023 total_%"] > base["2023 total_%"] else "  NON validé : l'amélioration ne se confirme pas en 2023")


if __name__ == "__main__":
    main()
