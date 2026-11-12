"""
research/backtest/real_wf_diagnostics.py

Pourquoi le moteur perd-il sur les paires réelles ? Sépare trois causes :
  1. aucun avantage brut (rendement sans coûts) ;
  2. coûts (comparaison brut / net) ;
  3. paires non stationnaires : barrière causale sur l'ADF du spread (fenêtre passée de 624 barres,
     recalculé toutes les 24 barres), appliquée aux ENTRÉES seulement.
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

from dataclasses import replace                                               # noqa: E402
from research.backtest.engine import BacktestConfig, BacktestEngine          # noqa: E402
from research.backtest.real_walk_forward import load_pair, bar_returns, summarize  # noqa: E402
from research.institutional.signal_engine import InstitutionalSignalEngine   # noqa: E402

GATES = (None, 0.10, 0.05, 0.01)
COSTS = {"brut (0 pb)": dict(fee_rate=0.0, slippage_bps=0.0), "net (5 pb/jambe)": dict(fee_rate=0.0004, slippage_bps=1.0)}


def causal_adf_gate(spreads: np.ndarray, start: int, window: int = 624, refresh: int = 24) -> np.ndarray:
    """p-valeur ADF du spread sur les `window` dernières barres (barre courante incluse), maintenue entre rafraîchissements."""
    from statsmodels.tsa.stattools import adfuller
    n = len(spreads)
    p = np.full(n, np.nan)
    last = np.nan
    for i in range(start, n):
        if (i - start) % refresh == 0:
            try:
                last = adfuller(spreads[i - window + 1:i + 1], maxlag=1, regression="c", autolag=None)[1]
            except Exception:
                last = np.nan
        p[i] = last
    return p


def run_pair(a: str, b: str):
    from loguru import logger
    logger.remove()
    df = load_pair(a, b)
    if len(df) < 2000:
        return None
    ts = df["timestamp"].to_numpy()
    sig = InstitutionalSignalEngine(solver_type="ergodic", cost=0.002).generate(ts, df["a"].to_numpy(float), df["b"].to_numpy(float))
    start = int(np.argmax(np.isfinite(sig.b_star)))
    pvals = causal_adf_gate(sig.spreads, max(start, 624))
    series, row = {}, {"pair": f"{a}x{b}"}
    for gate in GATES:
        if gate is None:
            gsig = sig
        else:
            ok = np.nan_to_num(pvals, nan=1.0) < gate
            gsig = replace(sig, entry_long=sig.entry_long * ok, entry_short=sig.entry_short * ok)
        for cname, ckw in COSTS.items():
            res = BacktestEngine(BacktestConfig(position_size=0.10, time_stop_mult=8.0, **ckw)).run(gsig)
            key = f"{'sans' if gate is None else 'ADF<' + str(gate)} | {cname}"
            row[f"ret[{key}]"] = res["metrics"]["total_return_pct"]
            row[f"trades[{key}]"] = res["metrics"]["n_trades"]
            series[key] = bar_returns(res["df"]).astype(np.float32)
    row["frac_adf_ok_05"] = float(np.nanmean(np.nan_to_num(pvals[max(start, 624):], nan=1.0) < 0.05))
    return row, ts, series


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "optimization_results", "real_wf_diagnostics"))
    args = ap.parse_args()
    pairs = pd.read_parquet(args.dataset, columns=["pair_a", "pair_b"]).drop_duplicates().values.tolist()
    if args.limit:
        pairs = pairs[:args.limit]
    rows = [r for r in Parallel(n_jobs=args.jobs, verbose=1)(delayed(run_pair)(a, b) for a, b in pairs) if r is not None]
    pd.DataFrame([r[0] for r in rows]).to_csv(args.out + "_pairs.csv", index=False)

    keys = list(rows[0][2].keys())
    out = []
    for k in keys:
        rets = pd.DataFrame({r[0]["pair"]: pd.Series(r[2][k], index=r[1]) for r in rows}).sort_index()
        s = summarize(rets, k)
        s["trades_med"] = float(np.median([r[0][f"trades[{k}]"] for r in rows]))
        s["pairs_win_%"] = float(np.mean([r[0][f"ret[{k}]"] > 0 for r in rows]) * 100)
        out.append(s)
    res = pd.DataFrame(out)
    res.to_csv(args.out + "_portfolio.csv", index=False)
    print(f"\n{len(rows)} paires ; part moyenne du temps où la barrière ADF<0.05 est ouverte : {np.mean([r[0]['frac_adf_ok_05'] for r in rows]):.0%}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
