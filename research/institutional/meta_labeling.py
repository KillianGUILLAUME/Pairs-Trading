"""
research/institutional/meta_labeling.py

Méta-label de la chaîne HJB -> XGB : définition UNIQUE du trade, du label, des features et de la décision,
partagée par l'entraînement (`universal_oracle_trainer.py`) et l'inférence (`run_generative_cvar.py`,
`backtester.py`, `performance_audit.py`).

  - événement : un TRADE du moteur HJB (politique à couverture verrouillée), pas une barre de la zone d'entrée ;
  - label     : 1 si le rendement NET de ce trade, tel que le backtester l'encaisse, est > 0. Le spread-signal
                n'est jamais utilisé comme gain : il revient vers zéro à chaque rafraîchissement de couverture
                sans gain encaissable (voir locked_policy.py) ;
  - features  : fenêtre de 128 barres finissant à la barre d'entrée, calculées sur le spread TRADÉ
                log A - beta_e log B - alpha_e. Aucune donnée extérieure à la paire : elles ont donc un sens sur
                une paire synthétique, pour laquelle BTC / ETH ne sont pas générés ;
  - décision  : garder ou rejeter le trade, une seule fois, à l'entrée (`BacktestEngine.run(keep_trade=...)`).

L'oracle sauvegardé porte l'empreinte de la chaîne sur laquelle il a été entraîné. `MetaOracle.load` refuse un
oracle sans empreinte (v2, v3, z-score), entraîné avec d'autres réglages, ou entraîné au-delà de la date évaluée.
"""

from __future__ import annotations

import os
import pickle
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from research.backtest.engine import BacktestConfig, BacktestEngine, CostModel
from research.backtest.real_walk_forward import DATA_DIR, PROJECT_ROOT, load_pair
from research.institutional.feature_master import FeatureMaster
from research.institutional.signal_engine import InstitutionalSignalEngine

ORACLE_PATH = os.path.join(PROJECT_ROOT, "data", "models", "institutional", "oracle_v4_hjb_locked.pkl")
SCHEMA = "hjb_meta_v1"
FEATURE_WINDOW = 128                                   # barres, barre d'entrée incluse
EMBARGO_MS = FEATURE_WINDOW * 3_600_000                # barres horaires
CENSORED = "fin de série"                              # trade encore ouvert : issue inconnue
META_COLUMNS = ("pair", "t_entry", "t_exit", "side", "kind", "net_ret", "label")

_ENGINE_KEYS = ("window", "solver_type", "rf", "cost", "use_kalman", "recalib_freq", "min_theta", "exit_mode",
                "fixed_beta", "mean_source", "hedge", "hedge_window", "hedge_refresh")
_POLICY_KEYS = ("fee_rate", "slippage_bps", "time_stop_mult", "time_stop_causal")


class OracleMismatch(RuntimeError):
    """L'oracle n'a pas été entraîné sur la chaîne (ou avant la date) à laquelle on veut l'appliquer."""


# ============================================================================
# Chaîne primaire (HJB)
# ============================================================================

def make_chain(solver_type: str = "ergodic", cost: float = 0.002) -> Tuple[InstitutionalSignalEngine, BacktestConfig]:
    """Moteur de signaux et réglages de la politique : les mêmes à l'entraînement et à l'inférence."""
    engine = InstitutionalSignalEngine(solver_type=solver_type, cost=cost)
    cfg = BacktestConfig(position_size=0.10, time_stop_mult=8.0, time_stop_causal=True)
    return engine, cfg


def chain_fingerprint(engine: InstitutionalSignalEngine, cfg: BacktestConfig) -> Dict[str, object]:
    fp: Dict[str, object] = {f"engine.{k}": getattr(engine, k) for k in _ENGINE_KEYS}
    fp.update({f"policy.{k}": getattr(cfg, k) for k in _POLICY_KEYS})
    return fp


def evaluation_start(engine: InstitutionalSignalEngine) -> int:
    """
    Première barre dont la fenêtre de calibration ne contient que des spreads HORS ÉCHANTILLON : avec la couverture
    OLS roulante, les `hedge_window` premières barres portent les résidus en échantillon du premier ajustement.
    """
    if engine.hedge == "rolling_ols" and engine.fixed_beta is None:
        return engine.window + engine.hedge_window
    return engine.window


# ============================================================================
# Label et features d'un trade
# ============================================================================

def trade_net_returns(res: dict, costs: CostModel) -> np.ndarray:
    """
    Rendement net de chaque trade de `BacktestEngine.run(lock_hedge=True)`, en fraction du notional par jambe :
    somme des (ret_a - beta_e ret_b) créditées par le moteur de gain, moins les coûts d'entrée et de sortie.
    """
    df = res["df"]
    spr = (df["ret_a"] - df["beta_r"] * df["ret_b"]).to_numpy(dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(spr)])      # cum[k] = somme des barres 0..k-1
    round_trip = 2.0 * costs.transaction_cost(1.0)     # entrée + sortie, deux jambes chacune
    return np.array([t["side"] * (cum[t["exit"] + 1] - cum[t["entry"] + 1]) - round_trip for t in res["trades"]])


def entry_features(fm: FeatureMaster, log_a: np.ndarray, log_b: np.ndarray, signal, i: int) -> Dict[str, float]:
    """Features à la barre d'entrée `i` (indice absolu dans log_a / log_b / signal), sur le spread tradé."""
    lo = i - FEATURE_WINDOW + 1
    if lo < 0:
        raise ValueError(f"Historique insuffisant à la barre {i} : {FEATURE_WINDOW} barres requises.")
    beta, alpha = float(signal.betas[i]), float(signal.alphas[i])
    feats = fm.get_feature_vector(log_a[lo:i + 1], log_b[lo:i + 1], beta=beta, alpha=alpha)
    x = float(log_a[i] - beta * log_b[i] - alpha)
    feats["spread_at_entry"] = x
    feats["dist_to_mu"] = abs(x - float(signal.mu_ref[i]))
    return feats


def feature_columns(data: pd.DataFrame) -> List[str]:
    return [c for c in data.columns if c not in META_COLUMNS]


def feature_matrix(data: pd.DataFrame, names: List[str]) -> np.ndarray:
    """
    Matrice (N, F) dans l'ordre de `names`. Une feature absente lève KeyError (jamais remplacée par 0) ; une valeur
    non finie devient NaN, que XGBoost traite nativement (une demi-vie infinie codée 0 se lisait « retour immédiat »).
    """
    X = data[list(names)].to_numpy(dtype=np.float64, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def build_pair_dataset(pair_a: str, pair_b: str, cost: float = 0.002) -> Optional[pd.DataFrame]:
    """Une ligne par trade CLOS de la chaîne HJB sur l'historique réel de la paire (None si historique insuffisant)."""
    missing = [p for p in (pair_a, pair_b) if not os.path.exists(os.path.join(DATA_DIR, f"{p}.parquet"))]
    if missing:
        logger.warning(f"{pair_a}x{pair_b} : parquet manquant pour {', '.join(missing)}, paire ignorée.")
        return None
    df = load_pair(pair_a, pair_b)
    engine, cfg = make_chain(cost=cost)
    start = evaluation_start(engine)
    if len(df) < start + 500:
        return None

    ts = df["timestamp"].to_numpy()
    pa, pb = df["a"].to_numpy(float), df["b"].to_numpy(float)
    log_a, log_b = np.log(pa), np.log(pb)
    signal = engine.generate(ts, pa, pb)
    bt = BacktestEngine(cfg)
    res = bt.run(signal.slice(start, len(df)), lock_hedge=True)

    fm = FeatureMaster(signature_level=2)
    rows = []
    for t, net in zip(res["trades"], trade_net_returns(res, bt.costs)):
        if t["kind"] == CENSORED:
            continue
        i, j = start + t["entry"], start + t["exit"]
        rows.append({"pair": f"{pair_a}x{pair_b}", "t_entry": int(ts[i]), "t_exit": int(ts[j]), "side": int(t["side"]),
                     "kind": t["kind"], "net_ret": float(net), "label": int(net > 0.0),
                     **entry_features(fm, log_a, log_b, signal, i)})
    return pd.DataFrame(rows) if rows else None


# ============================================================================
# Découpage temporel, calibration, seuil (sans sklearn ni xgboost)
# ============================================================================

def purged_walk_forward(t_entry, t_exit, n_splits: int = 5,
                        embargo_ms: int = EMBARGO_MS) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward à fenêtre croissante sur le temps CALENDAIRE, toutes paires confondues. Un trade n'est appris que
    s'il est CLOS plus de `embargo_ms` avant le début du bloc de validation : aucun label ne chevauche la validation.
    """
    t_entry, t_exit = np.asarray(t_entry), np.asarray(t_exit)
    edges = np.quantile(t_entry, np.linspace(0.0, 1.0, n_splits + 2))
    for k in range(1, n_splits + 1):
        lo, hi = edges[k], edges[k + 1]
        val = (t_entry >= lo) & ((t_entry <= hi) if k == n_splits else (t_entry < hi))
        train = t_exit < lo - embargo_ms
        if train.any() and val.any():
            yield np.flatnonzero(train), np.flatnonzero(val)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def apply_platt(raw, platt: Tuple[float, float]) -> np.ndarray:
    """Probabilité calibrée : sigmoïde(a logit(p) + b)."""
    a, b = platt
    return 1.0 / (1.0 + np.exp(-(a * logit(raw) + b)))


def select_threshold(p, net, min_kept: int = 50) -> float:
    """
    Seuil maximisant le rendement net TOTAL des trades gardés (prédictions hors échantillon). `inf` (tout rejeter)
    si aucun seuil ne donne un total positif : un filtre ne crée pas d'avantage là où il n'y en a pas.
    """
    p, net = np.asarray(p, dtype=float), np.asarray(net, dtype=float)
    best_thr, best_total = float("inf"), 0.0
    for thr in np.unique(np.quantile(p, np.linspace(0.0, 0.95, 20))):
        kept = p >= thr
        total = float(net[kept].sum())
        if kept.sum() >= min_kept and total > best_total:
            best_thr, best_total = float(thr), total
    return best_thr


# ============================================================================
# Oracle
# ============================================================================

class MetaOracle:
    """Oracle XGB (méta-label) chargé, avec sa calibration et son seuil choisis à l'entraînement."""

    def __init__(self, payload: dict):
        self.model = payload["model"]
        self.features = list(payload["features"])
        self.platt = tuple(payload["platt"])
        self.threshold = float(payload["threshold"])
        self.train_end_ms = int(payload["train_end_ms"])
        self._fm = FeatureMaster(signature_level=2)

    @classmethod
    def load(cls, path: str, engine: InstitutionalSignalEngine, cfg: BacktestConfig,
             as_of_ms: Optional[int] = None) -> "MetaOracle":
        """
        as_of_ms : début de la période évaluée. Un oracle entraîné au-delà l'a déjà vue : il est refusé.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} absent : lancer research/institutional/universal_oracle_trainer.py")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("schema") != SCHEMA:
            raise OracleMismatch(f"{os.path.basename(path)} : oracle sans empreinte de chaîne (entraîné sur un ancien "
                                 "signal), à réentraîner avec universal_oracle_trainer.py")
        expected, trained = chain_fingerprint(engine, cfg), payload["chain"]
        diff = [f"{k} = {trained.get(k)!r} à l'entraînement, {v!r} ici" for k, v in expected.items() if trained.get(k) != v]
        if diff:
            raise OracleMismatch(f"{os.path.basename(path)} : chaîne différente de l'entraînement ({' ; '.join(diff)})")
        if as_of_ms is not None and payload["train_end_ms"] > as_of_ms:
            raise OracleMismatch(
                f"{os.path.basename(path)} : entraîné jusqu'au {pd.Timestamp(payload['train_end_ms'], unit='ms'):%Y-%m-%d}, "
                f"après le début de la période évaluée ({pd.Timestamp(as_of_ms, unit='ms'):%Y-%m-%d})")
        return cls(payload)

    def proba(self, data: pd.DataFrame) -> np.ndarray:
        """Probabilité calibrée que chaque trade (ligne) soit gagnant net de coûts."""
        return apply_platt(self.model.predict_proba(feature_matrix(data, self.features))[:, 1], self.platt)

    def gate(self, log_a: np.ndarray, log_b: np.ndarray, signal, offset: int = 0) -> Callable[[dict], bool]:
        """
        Filtre `keep_trade` pour `BacktestEngine.run(signal.slice(offset, ...), lock_hedge=True)` : une décision par
        trade, à partir de la seule barre d'entrée. `offset` : début de la tranche dans log_a / log_b / signal.
        """
        def keep(trade: dict) -> bool:
            feats = entry_features(self._fm, log_a, log_b, signal, offset + trade["entry"])
            trade["p_win"] = float(self.proba(pd.DataFrame([feats]))[0])
            return trade["p_win"] >= self.threshold
        return keep
