"""
research/institutional/universal_oracle_trainer.py

Entraîne l'oracle XGB (méta-label) sur les trades de la chaîne HJB actuelle, toutes paires confondues.
Le trade, le label et les features sont définis dans `meta_labeling.py` (mêmes fonctions qu'à l'inférence).

Protocole, fixé avant les résultats comme dans le reste du dépôt :
  - trades CLOS avant `--cutoff` (2023-01-01) : apprentissage ; calibration et seuil choisis sur les prédictions
    hors échantillon d'un walk-forward purgé ;
  - trades OUVERTS à partir de `--cutoff` : évaluation, une seule fois, jamais utilisés pour choisir.
Un trade à cheval sur la coupure n'est ni appris ni évalué.

    python research/institutional/universal_oracle_trainer.py            # toutes les paires du dataset causal
    python research/institutional/universal_oracle_trainer.py --limit 40 # essai (~16 trades par paire et par an : il en faut
                                                                         # 300 hors échantillon, soit une vingtaine de paires)
"""

import os

# À fixer AVANT tout import numérique : torch (libiomp5), scikit-learn et XGBoost (libomp) chargent chacun leur
# runtime OpenMP dans le même processus (voir run_generative_cvar.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from loguru import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.institutional.meta_labeling import (          # noqa: E402
    ORACLE_PATH, SCHEMA, apply_platt, build_pair_dataset, chain_fingerprint, feature_columns, feature_matrix,
    logit, make_chain, purged_walk_forward, select_threshold,
)

MIN_FOLD_TRADES = 100      # en dessous, un pli n'apprend rien d'utile
MIN_OOF_TRADES = 300       # en dessous, calibration et seuil ne sont que du bruit


def make_model() -> xgb.XGBClassifier:
    """
    Modèle volontairement petit : l'événement est un trade (quelques milliers au total), plus une barre (des dizaines
    de milliers d'exemples quasi dupliqués). Pas de `scale_pos_weight`, qui décalibre les probabilités.
    """
    return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, min_child_weight=5, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", random_state=0, n_jobs=1)


def fit_platt(raw: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(logit(raw).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def out_of_fold(train: pd.DataFrame, names: List[str]) -> np.ndarray:
    """Scores bruts hors échantillon (NaN pour le premier bloc et les plis trop petits)."""
    X, y = feature_matrix(train, names), train["label"].to_numpy()
    oof = np.full(len(train), np.nan)
    for tr, va in purged_walk_forward(train["t_entry"].to_numpy(), train["t_exit"].to_numpy()):
        if len(tr) < MIN_FOLD_TRADES or len(np.unique(y[tr])) < 2:
            continue
        oof[va] = make_model().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


def report(label: str, trades: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    """Bilan net de coûts : tous les trades HJB contre ceux que l'oracle garde."""
    net = trades["net_ret"].to_numpy()
    kept = p >= threshold
    out = {"n": int(len(net)), "n_kept": int(kept.sum()), "threshold": float(threshold),
           "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
           "net_bp": float(net.mean() * 1e4) if len(net) else float("nan"),
           "win_rate_kept": float((net[kept] > 0).mean()) if kept.any() else float("nan"),
           "net_bp_kept": float(net[kept].mean() * 1e4) if kept.any() else float("nan")}
    logger.info(f"{label} : {out['n']} trades, gagnants {out['win_rate']:.1%}, net moyen {out['net_bp']:.0f} pb | "
                f"gardés {out['n_kept']}, gagnants {out['win_rate_kept']:.1%}, net moyen {out['net_bp_kept']:.0f} pb")
    return out


def train_universal_oracle(pairs: List[Tuple[str, str]], cutoff: str = "2023-01-01", out_path: str = ORACLE_PATH,
                           jobs: int = 6) -> dict:
    # 1. Un exemple par trade clos de la chaîne HJB
    frames = Parallel(n_jobs=jobs, verbose=1)(delayed(build_pair_dataset)(a, b) for a, b in pairs)
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("Aucun trade : vérifier les parquets et la longueur d'historique.")
    data = pd.concat(frames, ignore_index=True).sort_values("t_entry", kind="stable").reset_index(drop=True)
    names = feature_columns(data)

    # 2. Coupure chronologique
    cutoff_ms = int(pd.Timestamp(cutoff, tz="UTC").timestamp() * 1000)
    train = data[data["t_exit"] < cutoff_ms].reset_index(drop=True)
    test = data[data["t_entry"] >= cutoff_ms].reset_index(drop=True)
    logger.info(f"{len(data)} trades sur {data['pair'].nunique()} paires | apprentissage {len(train)} (clos avant {cutoff}) | "
                f"évaluation {len(test)} | à cheval, écartés {len(data) - len(train) - len(test)}")

    # 3. Calibration et seuil sur prédictions hors échantillon
    oof = out_of_fold(train, names)
    seen = np.isfinite(oof)
    y, net = train["label"].to_numpy(), train["net_ret"].to_numpy()
    if seen.sum() < MIN_OOF_TRADES or len(np.unique(y[seen])) < 2:
        raise RuntimeError(f"{int(seen.sum())} prédictions hors échantillon exploitables ({MIN_OOF_TRADES} requises) : "
                           "trop peu de trades pour entraîner un méta-label.")
    platt = fit_platt(oof[seen], y[seen])
    p_oof = apply_platt(oof[seen], platt)
    if platt[0] > 0:
        threshold = select_threshold(p_oof, net[seen])
    else:
        threshold = float("inf")
        logger.warning(f"Pente de calibration {platt[0]:.2f} <= 0 : le score n'ordonne pas les trades, l'oracle rejette tout.")
    if not np.isfinite(threshold):
        logger.warning("Aucun seuil ne donne un rendement net positif hors échantillon : l'oracle rejette tous les trades.")

    # 4. Modèle final, puis évaluation unique
    model = make_model().fit(feature_matrix(train, names), y)
    reports = {"oof": report("Hors échantillon (apprentissage)", train[seen], p_oof, threshold)}
    if len(test):
        p_test = apply_platt(model.predict_proba(feature_matrix(test, names))[:, 1], platt)
        reports["test"] = report(f"Évaluation (à partir de {cutoff})", test, p_test, threshold)

    # 5. Sauvegarde : le modèle, et tout ce qui permet de refuser un mauvais usage
    engine, cfg = make_chain()
    payload = {"schema": SCHEMA, "model": model, "features": names, "platt": platt, "threshold": threshold,
               "chain": chain_fingerprint(engine, cfg), "train_end_ms": cutoff_ms, "reports": reports}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Oracle sauvegardé dans {out_path}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2023-01-01", help="date UTC séparant apprentissage et évaluation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=ORACLE_PATH)
    ap.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "data", "storage", "screened", "super_dataset_SDE_128_causal.parquet"))
    args = ap.parse_args()
    pairs = pd.read_parquet(args.dataset, columns=["pair_a", "pair_b"]).drop_duplicates().values.tolist()
    if args.limit:
        pairs = pairs[:args.limit]
    train_universal_oracle([tuple(p) for p in pairs], cutoff=args.cutoff, out_path=args.out, jobs=args.jobs)


if __name__ == "__main__":
    main()
