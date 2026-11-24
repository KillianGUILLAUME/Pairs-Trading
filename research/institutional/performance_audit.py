"""
research/institutional/performance_audit.py

Bilan de l'oracle XGB (méta-label) sur quelques paires, HORS ÉCHANTILLON : uniquement les trades ouverts après la
fin de son apprentissage (`train_end_ms`), net de coûts. L'ancienne version évaluait les 20 % finaux d'un jeu sur
lequel l'oracle avait été entraîné en entier.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # avant tout import numérique (voir run_generative_cvar.py)

import sys

import pandas as pd
from loguru import logger

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.meta_labeling import ORACLE_PATH, MetaOracle, build_pair_dataset, make_chain  # noqa: E402

AUDIT_PAIRS = [("ADA_USDT", "AVAX_USDT"), ("LINK_USDT", "SOL_USDT"), ("ETH_USDT", "BTC_USDT"), ("BNB_USDT", "NEAR_USDT")]


def _pct(num: int, den: int) -> str:
    return f"{num / den:.2%}" if den else "N/A"


def run_performance_audit():
    engine, cfg = make_chain()
    oracle = MetaOracle.load(ORACLE_PATH, engine, cfg)
    since = f"{pd.Timestamp(oracle.train_end_ms, unit='ms'):%Y-%m-%d}"
    logger.info(f"Audit de l'oracle : trades ouverts depuis le {since} (fin de l'apprentissage)")

    frames = [build_pair_dataset(a, b) for a, b in AUDIT_PAIRS]
    frames = [f for f in frames if f is not None]
    if not frames:
        logger.error("Aucun trade sur les paires d'audit.")
        return
    data = pd.concat(frames, ignore_index=True)
    oos = data[data["t_entry"] >= oracle.train_end_ms].reset_index(drop=True)
    if oos.empty:
        logger.error(f"Aucun trade ouvert depuis le {since} : rien à évaluer hors échantillon.")
        return

    kept = oracle.proba(oos) >= oracle.threshold
    win = (oos["label"] == 1).to_numpy()
    net = oos["net_ret"].to_numpy()
    n, n_kept = len(oos), int(kept.sum())

    print("\n" + "=" * 55)
    print(f"BILAN HORS ÉCHANTILLON (seuil {oracle.threshold:.2f})")
    print("=" * 55)
    print("1. SIGNAL HJB :")
    print(f"   - Trades :                  {n}")
    print(f"   - Gagnants nets :           {int(win.sum())} ({_pct(int(win.sum()), n)})")
    print(f"   - Rendement net moyen :     {net.mean() * 1e4:.0f} pb par trade")
    print("\n2. APRÈS FILTRE XGB :")
    print(f"   - Trades gardés :           {n_kept} ({_pct(n_kept, n)})")
    print(f"   - Gagnants nets :           {_pct(int((kept & win).sum()), n_kept)}")
    print("   - Rendement net moyen :     " + (f"{net[kept].mean() * 1e4:.0f} pb par trade" if n_kept else "N/A (0 trade)"))
    print("\n3. COÛT D'OPPORTUNITÉ :")
    print(f"   - Gagnants rejetés :        {int((~kept & win).sum())} ({_pct(int((~kept & win).sum()), int(win.sum()))} des gagnants)")

    importance = pd.Series(oracle.model.feature_importances_, index=oracle.features).sort_values(ascending=False)
    print("\n" + "=" * 55)
    print("FACTEURS DE DÉCISION (importance XGB)")
    print("=" * 55)
    print(importance.head(5))


if __name__ == "__main__":
    run_performance_audit()
