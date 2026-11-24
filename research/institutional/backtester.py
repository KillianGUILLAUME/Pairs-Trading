"""
research/institutional/backtester.py

Chaîne HJB (+ oracle XGB) sur l'historique RÉEL d'une paire, pour les deux solveurs.

  - politique à couverture verrouillée, arrêt temporel causal (`meta_labeling.make_chain`) ;
  - l'oracle ne sert que s'il a été entraîné sur la chaîne évaluée, et seulement sur la période POSTÉRIEURE à son
    apprentissage ; HJB seul et HJB + XGB sont comparés sur cette même période ;
  - le coût de portage estimé entre dans le capital, donc dans les métriques.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # avant tout import numérique (voir run_generative_cvar.py)

import sys

import numpy as np
import pandas as pd
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.backtest.engine import BacktestEngine, PerformanceMetrics                         # noqa: E402
from research.backtest.real_walk_forward import load_pair                                       # noqa: E402
from research.institutional.funding_estimator import FundingEstimator                           # noqa: E402
from research.institutional.meta_labeling import (                                              # noqa: E402
    ORACLE_PATH, MetaOracle, OracleMismatch, evaluation_start, make_chain,
)

MIN_BARS = 500      # période évaluée minimale


def apply_carry(df_pnl: pd.DataFrame, funding_a: np.ndarray, funding_b: np.ndarray, initial_capital: float) -> pd.DataFrame:
    """
    Portage de la position TENUE pendant la barre (décidée à la barre précédente), sur le notional par jambe du trade,
    jambe B pondérée par beta : un long A / short B paie le funding de A et reçoit beta x celui de B.
    Le capital est recalculé : c'est sur lui que `PerformanceMetrics` lit rendement, Sharpe et drawdown.
    """
    out = df_pnl.copy()
    held = out["position"].shift(1).fillna(0.0).to_numpy()
    notional = out["notional"].shift(1).fillna(0.0).to_numpy()
    carry = held * notional * (out["beta_r"].to_numpy() * funding_b - funding_a)
    out["net_pnl"] = out["net_pnl"].to_numpy() + carry
    out["capital"] = initial_capital + out["net_pnl"].cumsum()
    return out


def run_institutional_comparison(pair_a: str, pair_b: str, oracle_path: str = ORACLE_PATH) -> dict:
    logger.info(f"Analyse institutionnelle : {pair_a} x {pair_b}")
    df = load_pair(pair_a, pair_b)
    ts = df["timestamp"].to_numpy()
    pa, pb = df["a"].to_numpy(float), df["b"].to_numpy(float)
    log_a, log_b = np.log(pa), np.log(pb)

    # Funding ESTIMÉ (modèle empirique, pas des taux réels) ; NaN sur la première fenêtre glissante
    fe = FundingEstimator()
    funding_a = np.nan_to_num(fe.estimate_funding_series(pa))
    funding_b = np.nan_to_num(fe.estimate_funding_series(pb))

    results = {}
    for solver_name in ("ergodic", "fpt"):
        engine, cfg = make_chain(solver_type=solver_name)
        start = evaluation_start(engine)
        try:
            oracle = MetaOracle.load(oracle_path, engine, cfg)
            start = max(start, int(np.searchsorted(ts, oracle.train_end_ms)))
        except (FileNotFoundError, OracleMismatch) as e:
            logger.warning(f"{solver_name} : XGB désactivé, HJB seul ({e})")
            oracle = None
        if len(ts) - start < MIN_BARS:
            logger.error(f"{solver_name} : {max(len(ts) - start, 0)} barres à évaluer, {MIN_BARS} requises.")
            continue

        signal = engine.generate(ts, pa, pb)
        ahead = signal.slice(start, len(ts))
        bt = BacktestEngine(cfg)
        runs = {"HJB seul": bt.run(ahead, lock_hedge=True)}
        if oracle is not None:
            runs["HJB + XGB"] = bt.run(ahead, lock_hedge=True, keep_trade=oracle.gate(log_a, log_b, signal, offset=start))
            kept = sum(t["kept"] for t in runs["HJB + XGB"]["trades"])
            logger.info(f"{solver_name} : XGB garde {kept} trades sur {len(runs['HJB + XGB']['trades'])}")

        for name, res in runs.items():
            df_pnl = apply_carry(res["df"], funding_a[start:], funding_b[start:], cfg.initial_capital)
            results[f"{solver_name} | {name}"] = {"metrics": PerformanceMetrics.compute(df_pnl), "signal": signal}

    logger.info("=" * 60)
    for name, r in results.items():
        m = r["metrics"]
        logger.info(f"{name:<22} | Sharpe {m['sharpe']:6.3f} | rendement {m['total_return_pct']:7.2f}% | "
                    f"MDD {m['max_dd']:6.2f}% | trades {m['n_trades']}")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    run_institutional_comparison("ADA_USDT", "AVAX_USDT")
