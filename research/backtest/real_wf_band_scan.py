"""
research/backtest/real_wf_band_scan.py

Échelle des bandes k choisie sur 2022 puis évaluée sur 2023 (un seul paramètre, split chronologique).
Le moteur est causal ; k multiplie les écarts des seuils à μ (entrées et sorties). k = 1 : seuils de la théorie.
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
from research.backtest.model_misspecification import rescale_signal          # noqa: E402
from research.backtest.real_walk_forward import load_pair, bar_returns, summarize  # noqa: E402
from research.institutional.signal_engine import InstitutionalSignalEngine   # noqa: E402

K_GRID = (1.0, 1.5, 2.0, 3.0, 4.0)
COSTS = {"brut": dict(fee_rate=0.0, slippage_bps=0.0), "net": dict(fee_rate=0.0004, slippage_bps=1.0)}


def run_pair(a: str, b: str, exit_mode: str = "optimal"):
    from loguru import logger
    logger.remove()
    df = load_pair(a, b)
    if len(df) < 2000:
        return None
    ts = df["timestamp"].to_numpy()
    sig = InstitutionalSignalEngine(solver_type="ergodic", cost=0.002, exit_mode=exit_mode).generate(ts, df["a"].to_numpy(float), df["b"].to_numpy(float))
    series, trades = {}, {}
    for k in K_GRID:
        ks = rescale_signal(sig, k)
        for cname, ckw in COSTS.items():
            res = BacktestEngine(BacktestConfig(position_size=0.10, time_stop_mult=8.0, **ckw)).run(ks)
            series[(k, cname)] = bar_returns(res["df"]).astype(np.float32)
            if cname == "net":
                trades[k] = res["metrics"]["n_trades"]
    return f"{a}x{b}", ts, series, trades


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exit-mode", default="optimal", choices=["optimal", "mean", "flip"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    args = ap.parse_args()
    pairs = pd.read_parquet(args.dataset, columns=["pair_a", "pair_b"]).drop_duplicates().values.tolist()
    if args.limit:
        pairs = pairs[:args.limit]
    rows = [r for r in Parallel(n_jobs=args.jobs, verbose=1)(delayed(run_pair)(a, b, args.exit_mode) for a, b in pairs) if r is not None]

    out = []
    for k in K_GRID:
        for cname in COSTS:
            rets = pd.DataFrame({r[0]: pd.Series(r[2][(k, cname)], index=r[1]) for r in rows}).sort_index()
            year = pd.to_datetime(rets.index, unit="ms").year
            s22 = summarize(rets[year == 2022], "")
            s23 = summarize(rets[year == 2023], "")
            out.append({"k": k, "coûts": cname,
                        "2022 total_%": s22["total_%"], "2022 sharpe": s22["sharpe"],
                        "2023 total_%": s23["total_%"], "2023 sharpe": s23["sharpe"],
                        "trades médian": float(np.median([r[3][k] for r in rows]))})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(PROJECT_ROOT, "data", "optimization_results", f"real_wf_band_scan_{args.exit_mode}.csv"), index=False)
    print(f"\n{len(rows)} paires, sortie={args.exit_mode}, portefeuille équipondéré (k choisi sur 2022, évalué sur 2023)")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
