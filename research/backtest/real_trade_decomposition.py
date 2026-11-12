"""
research/backtest/real_trade_decomposition.py

Où la perte se forme-t-elle ? Pour chaque trade (segment de position constante non nulle) :
  sig_move   = côté x (x_sortie - x_entrée)           mouvement du spread-signal (celui sur lequel on décide)
  pnl_unit   = côté x somme (ret_a - beta_r ret_b)     gain réellement encaissé par unité de notional (brut)
  mismatch   = pnl_unit - sig_move                     écart signal / gain (sauts de couverture, bêta décalé, convexité)
  band       = a* sigma_inf à l'entrée (demi-largeur de bande) ; la théorie promet sig_move ~ 2 a* si le trade se termine à l'autre bande
  exit_kind  = retournement | signal de sortie | arrêt temporel | fin de série
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
from research.backtest.real_walk_forward import load_pair                      # noqa: E402
from research.institutional.signal_engine import InstitutionalSignalEngine   # noqa: E402


def pair_trades(a: str, b: str, exit_mode: str, hedge_refresh: int = 24, lock: bool = False):
    from loguru import logger
    logger.remove()
    df = load_pair(a, b)
    if len(df) < 2000:
        return None
    ts = df["timestamp"].to_numpy()
    sig = InstitutionalSignalEngine(solver_type="ergodic", cost=0.002, exit_mode=exit_mode, hedge_refresh=hedge_refresh).generate(
        ts, df["a"].to_numpy(float), df["b"].to_numpy(float))
    cfg = BacktestConfig(position_size=0.10, time_stop_mult=8.0, fee_rate=0.0, slippage_bps=0.0)
    res = BacktestEngine(cfg).run(sig, lock_hedge=lock)
    la, lb = np.log(df['a'].to_numpy(float)), np.log(df['b'].to_numpy(float))
    pos = res["positions"].astype(int)
    bt = res["df"]
    spr_ret = (bt["ret_a"].to_numpy() - bt["beta_r"].to_numpy() * bt["ret_b"].to_numpy())
    x = sig.spreads
    n = len(pos)
    rows = []
    i = 1
    while i < n:
        if pos[i] != 0 and pos[i] != pos[i - 1]:
            side, entry = pos[i], i
            j = i
            while j + 1 < n and pos[j + 1] == side:
                j += 1
            exit_bar = j + 1 if j + 1 < n else None          # barre où la position change
            last = exit_bar if exit_bar is not None else j     # dernière barre créditée
            if exit_bar is None:
                kind = "fin de série"
            elif pos[exit_bar] == -side:
                kind = "retournement"
            else:
                ex = (sig.exit_long if side == 1 else sig.exit_short)
                kind = "signal de sortie" if (ex is not None and ex[exit_bar] > 0) else "arrêt temporel"
            band = float(sig.b_star[entry] - sig.mu_ref[entry]) if np.isfinite(sig.b_star[entry]) else np.nan
            rows.append({
                "pair": f"{a}x{b}", "year": int(pd.to_datetime(ts[entry], unit="ms").year), "side": int(side),
                "duration": int(last - entry), "kind": kind, "band": band,
                "sig_move_apparent": float(side * (x[last] - x[entry])),
                "sig_move": float(side * ((la[last] - sig.betas[entry] * lb[last] - sig.alphas[entry]) - (la[entry] - sig.betas[entry] * lb[entry] - sig.alphas[entry]))),
                "pnl_unit": float(side * np.sum(spr_ret[entry + 1:last + 1])),
                "cycle_theory": float(sig.cycle_bars[entry]) if sig.cycle_bars is not None else np.nan,
            })
            i = last if exit_bar is None else exit_bar
            if exit_bar is not None and pos[exit_bar] == -side:
                continue
        i += 1
    return rows


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hedge-refresh", type=int, default=24)
    ap.add_argument("--lock", action="store_true", help="couverture verrouillée à l'entrée")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--exit-mode", default="optimal", choices=["optimal", "mean", "flip"])
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    args = ap.parse_args()
    pairs = pd.read_parquet(args.dataset, columns=["pair_a", "pair_b"]).drop_duplicates().values.tolist()
    if args.limit:
        pairs = pairs[:args.limit]
    allrows = []
    for r in Parallel(n_jobs=args.jobs, verbose=1)(delayed(pair_trades)(a, b, args.exit_mode, args.hedge_refresh, args.lock) for a, b in pairs):
        if r:
            allrows += r
    t = pd.DataFrame(allrows)
    t["mismatch"] = t.pnl_unit - t.sig_move
    t["move_over_2a"] = t.sig_move / (2 * t.band)
    t.to_csv(os.path.join(PROJECT_ROOT, "data", "optimization_results", f"real_trades_{args.exit_mode}_refresh{args.hedge_refresh}{'_locked' if args.lock else ''}.csv"), index=False)

    print(f"\n{len(t)} trades sur {t.pair.nunique()} paires, sortie={args.exit_mode} ; unités : fraction du notional (bp = x1e4)")
    g = t.groupby("kind").agg(n=("pair", "size"), part=("pair", lambda s: len(s) / len(t)), duree_med=("duration", "median"),
                              sig_move_bp=("sig_move", lambda s: s.mean() * 1e4), pnl_bp=("pnl_unit", lambda s: s.mean() * 1e4),
                              mismatch_bp=("mismatch", lambda s: s.mean() * 1e4), gagnants=("pnl_unit", lambda s: (s > 0).mean()))
    print(g.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nmouvement du spread-signal COURANT (apparent) moyen : {t.sig_move_apparent.mean()*1e4:.0f} bp | mouvement du spread à couverture ENTRÉE : {t.sig_move.mean()*1e4:.0f} bp")
    print(f"Tous trades : mouvement signal moyen {t.sig_move.mean()*1e4:.0f} bp | gain réalisé moyen {t.pnl_unit.mean()*1e4:.0f} bp | écart moyen {t.mismatch.mean()*1e4:.0f} bp")
    print(f"largeur de bande moyenne 2a* : {2*t.band.mean()*1e4:.0f} bp | trades dont le mouvement signal atteint >= 80 % de 2a* : {(t.move_over_2a >= 0.8).mean():.0%}")
    print(f"durée médiane réalisée {t.duration.median():.0f} barres | cycle théorique médian {t.cycle_theory.median():.0f} barres")
    print(f"corrélation(mouvement signal, gain réalisé) : {np.corrcoef(t.sig_move, t.pnl_unit)[0,1]:.2f} | écart-type du mismatch {t.mismatch.std()*1e4:.0f} bp contre écart-type du mouvement {t.sig_move.std()*1e4:.0f} bp")
    for y in (2022, 2023):
        s = t[t.year == y]
        print(f"{y}: {len(s)} trades, gain brut moyen {s.pnl_unit.mean()*1e4:.0f} bp, gagnants {(s.pnl_unit>0).mean():.0%}")


if __name__ == "__main__":
    main()
