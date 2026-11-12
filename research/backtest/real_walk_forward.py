"""
research/backtest/real_walk_forward.py

Évaluation hors échantillon du moteur HJB sur les paires RÉELLES.

Le moteur est causal : à la barre i, couverture OLS, calibration AR(1), seuils et signaux
n'utilisent que les barres <= i. Les réglages (fenêtre 624, OLS 500 barres, solveur ergodique,
arrêt temporel 8 x demi-vie) ont été choisis sur SIMULATION, pas sur ces données : la période
2022-2023 est donc hors échantillon pour eux.

Chaque paire dispose de sa propre tranche de capital (position = 10 % du capital par jambe) ;
le portefeuille est l'équipondération des tranches (une tranche sans signal reste au repos).
Sensibilité aux coûts : mêmes signaux, backtest rejoué avec des frais plus élevés.

    python research/backtest/real_walk_forward.py            # toutes les paires du dataset causal
    python research/backtest/real_walk_forward.py --limit 6  # essai rapide
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
from research.institutional.signal_engine import InstitutionalSignalEngine   # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "storage", "parquet", "1h")
BARS_PER_YEAR = 8760.0

COST_SCENARIOS = {
    "base (5 pb/jambe)": dict(fee_rate=0.0004, slippage_bps=1.0),
    "x2 (10 pb/jambe)": dict(fee_rate=0.0008, slippage_bps=2.0),
    "x4 (20 pb/jambe)": dict(fee_rate=0.0016, slippage_bps=4.0),
}


def load_pair(a: str, b: str) -> pd.DataFrame:
    da = pd.read_parquet(os.path.join(DATA_DIR, f"{a}.parquet"))[["timestamp", "close"]].rename(columns={"close": "a"})
    db = pd.read_parquet(os.path.join(DATA_DIR, f"{b}.parquet"))[["timestamp", "close"]].rename(columns={"close": "b"})
    return da.merge(db, on="timestamp").sort_values("timestamp").reset_index(drop=True)


def bar_returns(df_bt: pd.DataFrame) -> np.ndarray:
    eq = df_bt["capital"].to_numpy()
    r = np.zeros(len(eq))
    r[1:] = np.diff(eq) / eq[:-1]
    return r


def run_pair(a: str, b: str, config: str):
    from loguru import logger
    logger.remove()
    df = load_pair(a, b)
    if len(df) < 2000:
        return None
    ts = df["timestamp"].to_numpy()
    pa, pb = df["a"].to_numpy(float), df["b"].to_numpy(float)

    if config == "nouveau":
        eng = InstitutionalSignalEngine(solver_type="ergodic", cost=0.002)
        time_stop = 8.0
    else:  # configuration historique : Kalman, seuils sur EMA rapide, fenêtre 156
        eng = InstitutionalSignalEngine(window=156, solver_type="ergodic", cost=0.002, use_kalman=True,
                                        mean_source="ema", hedge="kalman")
        time_stop = 2.0
    sig = eng.generate(ts, pa, pb)

    out = {"pair": f"{a}x{b}", "config": config, "n_bars": len(df), "first_valid": int(np.argmax(np.isfinite(sig.b_star)))}
    series = {}
    for name, cost_kw in COST_SCENARIOS.items():
        cfg = BacktestConfig(position_size=0.10, time_stop_mult=time_stop, **cost_kw)
        res = BacktestEngine(cfg).run(sig, lock_hedge=(config == "nouveau"))   # nouveau : couverture verrouillée à l'entrée
        m = res["metrics"]
        out[f"ret[{name}]"] = m["total_return_pct"]
        if name.startswith("base"):
            out.update({"sharpe": m["sharpe"], "max_dd": m["max_dd"], "n_trades": m["n_trades"], "in_mkt": m["pct_in_market"]})
        series[name] = bar_returns(res["df"]).astype(np.float32)
    out["diag_failures"] = sig.diagnostics["n_solver_failures"]
    out["diag_at_bound"] = sig.diagnostics["n_at_bound"]
    return out, ts, series


def summarize(rets: pd.DataFrame, label: str) -> dict:
    port = rets.fillna(0.0).mean(axis=1)
    eq = (1.0 + port).cumprod()
    sharpe = port.mean() / port.std() * np.sqrt(BARS_PER_YEAR) if port.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(port) / BARS_PER_YEAR
    return {"label": label, "total_%": (eq.iloc[-1] - 1) * 100, "annual_%": (eq.iloc[-1] ** (1 / years) - 1) * 100,
            "sharpe": sharpe, "max_dd_%": dd * 100}


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "optimization_results", "real_walk_forward"))
    args = ap.parse_args()

    pairs = pd.read_parquet(args.dataset, columns=["pair_a", "pair_b"]).drop_duplicates().values.tolist()
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} paires, deux configurations, {len(COST_SCENARIOS)} scénarios de coûts")

    results = {}
    for config in ("nouveau", "historique"):
        rows = Parallel(n_jobs=args.jobs, verbose=1)(delayed(run_pair)(a, b, config) for a, b in pairs)
        results[config] = [r for r in rows if r is not None]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary_rows = []
    for config, rows in results.items():
        df_pairs = pd.DataFrame([r[0] for r in rows])
        df_pairs.to_csv(f"{args.out}_{config}_pairs.csv", index=False)
        print(f"\n=== Configuration {config} : {len(df_pairs)} paires ===")
        base = df_pairs["ret[base (5 pb/jambe)]"]
        print(f"rendement par paire (tranche de capital, base) : médiane {base.median():.2f}% | moyenne {base.mean():.2f}% | "
              f"paires gagnantes {(base > 0).mean():.0%} | 10e centile {base.quantile(.1):.2f}% | 90e centile {base.quantile(.9):.2f}%")
        print(f"Sharpe par paire médian {df_pairs.sharpe.median():.2f} | trades médian {df_pairs.n_trades.median():.0f} | temps en marché médian {df_pairs.in_mkt.median():.0f}%")
        print(f"échecs du solveur (total) {int(df_pairs.diag_failures.sum())} | butées de grille (total) {int(df_pairs.diag_at_bound.sum())}")

        for scen in COST_SCENARIOS:
            cols = {}
            for out, ts, series in rows:
                cols[out["pair"]] = pd.Series(series[scen], index=ts)
            rets = pd.DataFrame(cols).sort_index()
            s = summarize(rets, f"{config} | {scen}")
            summary_rows.append(s)
            if scen.startswith("base"):
                idx = pd.to_datetime(rets.index, unit="ms")
                for label, mask in (("2022", idx.year == 2022), ("2023", idx.year == 2023)):
                    sub = summarize(rets[mask], f"{config} | {scen} | {label}")
                    summary_rows.append(sub)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(f"{args.out}_portfolio.csv", index=False)
    print("\n=== Portefeuille équipondéré (153 tranches de capital) ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
