"""
research/backtest/run_generative_cvar.py

Chaîne générateur -> HJB -> XGB, en distribution :

  1. historique RÉEL de la paire jusqu'à un point d'ancrage : c'est lui qui calibre la couverture, l'OU et les seuils ;
  2. générateur : `n_paths` prolongements synthétiques ancrés sur la dernière barre réelle, de la longueur
     d'entraînement du générateur par défaut (au-delà : extrapolation non validée) ;
  3. moteur HJB sur [historique réel + prolongement], à plat à l'ancrage ; seul le prolongement est évalué ;
  4. oracle XGB (méta-label) : garde ou rejette CHAQUE trade du moteur, une fois, à l'entrée. Sans oracle entraîné
     sur cette chaîne et avant l'ancrage, l'étape est sautée et signalée.

Le résultat est la dispersion du gain SOUS LA LOI DU GÉNÉRATEUR. Il ne valide ni le moteur ni l'oracle : un filtre qui
« marche » sur des trajectoires synthétiques mesure le générateur. Le jugement de l'oracle se fait sur trades réels hors
échantillon (universal_oracle_trainer.py, backtester.py).

    python research/backtest/run_generative_cvar.py --pair_a ADA_USDT --pair_b AVAX_USDT --n_paths 200
"""

import os

# torch (libiomp5), scikit-learn et XGBoost (libomp) chargent chacun leur runtime OpenMP. Observé sur Mac : avec les
# trois dans le processus, la génération se bloque à 0 % de CPU ; avec un seul thread, 20 trajectoires de 2000 barres
# prennent 5 s. À fixer AVANT tout import numérique.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.backtest.engine import BacktestEngine                                    # noqa: E402
from research.backtest.generative_sde_engine import GenerativeSDEEngine                # noqa: E402
from research.institutional.meta_labeling import (                                     # noqa: E402
    ORACLE_PATH, MetaOracle, OracleMismatch, evaluation_start, make_chain,
)

PATHS_DIR = os.path.join(project_root, "data", "storage", "synthetic_paths")
MAX_PRICE_RATIO = 1e3      # au-delà, la trajectoire a divergé : rejetée et comptée


def load_aligned(pair_a: str, pair_b: str) -> pd.DataFrame:
    """Clôtures (a, b, btc, eth) alignées PAR TIMESTAMP : intersection des quatre séries."""
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")

    def close(symbol: str) -> pd.Series:
        return pd.read_parquet(os.path.join(data_dir, f"{symbol}.parquet")).set_index("timestamp")["close"].astype(float)

    series = {"a": close(pair_a), "b": close(pair_b), "btc": close("BTC_USDT"), "eth": close("ETH_USDT")}
    return pd.concat(series, axis=1, join="inner").dropna().sort_index()


def latest_checkpoint() -> str:
    models_dir = os.path.join(project_root, "data", "models")
    names = sorted(f for f in os.listdir(models_dir) if f.startswith("neural_sde_") and f.endswith("world_model.pt"))
    if not names:
        raise FileNotFoundError(f"Aucun checkpoint neural_sde_*world_model.pt dans {models_dir}")
    return os.path.join(models_dir, names[-1])       # noms horodatés : le dernier est le plus récent


def synthetic_markets(sde: GenerativeSDEEngine, label: str, aligned: pd.DataFrame, anchor: int, horizon: int,
                      n_paths: int, seed: int) -> list:
    """
    `n_paths` prolongements de `horizon` barres (plus la barre d'ancrage, indice 0), en cache. La clé contient tout ce
    qui change les trajectoires : ancrage (donc prix initiaux et conditions), horizon, graine et checkpoint.
    """
    stem = os.path.splitext(os.path.basename(sde.model_path))[0]
    path = os.path.join(PATHS_DIR, f"{label}_n{n_paths}_h{horizon}_s{seed}_t{int(aligned.index[anchor])}_{stem}.parquet")
    if os.path.exists(path):
        logger.info(f"Trajectoires en cache : {path}")
        return [g for _, g in pd.read_parquet(path).groupby(level=0)]

    conditions = None
    if sde.conditioning is not None:
        hist = np.log(aligned.iloc[:anchor + 1])     # se termine à l'ancrage, rien après
        conditions = sde.conditions_from_history(*(hist[c].to_numpy() for c in ("a", "b", "btc", "eth")))
    else:
        logger.warning("Checkpoint sans métadonnées de conditionnement : conditions nulles (fenêtre moyenne).")
    markets = sde.simulate_markets(n_paths=n_paths, bars=horizon + 1, p_a_init=float(aligned["a"].iloc[anchor]),
                                   p_b_init=float(aligned["b"].iloc[anchor]), conditions=conditions, seed=seed)
    os.makedirs(PATHS_DIR, exist_ok=True)
    pd.concat(markets, keys=range(n_paths)).to_parquet(path)
    logger.info(f"Trajectoires mises en cache : {path}")
    return markets


def diverged(price: np.ndarray) -> bool:
    return not np.all(np.isfinite(price)) or price.min() <= 0 or price.max() / price.min() > MAX_PRICE_RATIO


def summarize(metrics: pd.DataFrame) -> dict:
    ret = metrics["total_return_pct"]
    var = np.percentile(ret, 5)
    return {
        "Sharpe moyen": metrics["sharpe"].mean(),
        "PnL moyen (%)": ret.mean(),
        "Drawdown moyen (%)": metrics["max_dd"].mean(),
        "VaR 95% PnL (%)": var,
        "CVaR 95% PnL (%)": ret[ret <= var].mean(),
        "Drawdown, 5e centile (%)": np.percentile(metrics["max_dd"], 5),
        "Trades par trajectoire": metrics["n_trades"].mean(),
    }


def plot_distributions(runs: dict, title: str, path: str) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for (name, m), color in zip(runs.items(), ("red", "cyan")):
        s = summarize(m)
        sns.kdeplot(m["total_return_pct"], ax=axes[0], color=color, label=name, fill=True, alpha=0.3, warn_singular=False)
        axes[0].axvline(s["CVaR 95% PnL (%)"], color=color, linestyle=":")
        sns.kdeplot(m["max_dd"], ax=axes[1], color=color, label=name, fill=True, alpha=0.3, warn_singular=False)
        axes[1].axvline(s["Drawdown, 5e centile (%)"], color=color, linestyle=":")
    axes[0].axvline(0, color="white", linestyle="--", linewidth=1)
    axes[0].set_title(f"Distribution générative du PnL (%) - {title}")
    axes[1].set_title("Distribution générative du drawdown (%)")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Graphique sauvegardé : {path}")


def run_generative_cvar(pair_a: str, pair_b: str, n_paths: int = 200, horizon: int | None = None,
                        anchor_offset: int = 0, seed: int = 0, oracle_path: str = ORACLE_PATH) -> pd.DataFrame | None:
    engine, bt_cfg = make_chain()
    bt = BacktestEngine(bt_cfg)
    warm = evaluation_start(engine)

    # 1. Historique réel et point d'ancrage
    aligned = load_aligned(pair_a, pair_b)
    anchor = len(aligned) - 1 - anchor_offset
    if anchor_offset < 0 or anchor < warm:
        raise ValueError(f"Ancrage impossible : {warm + 1} barres d'historique commun requises, "
                         f"{max(anchor + 1, 0)} disponibles (anchor_offset={anchor_offset}).")
    anchor_ts = int(aligned.index[anchor])
    logger.info(f"{pair_a} x {pair_b} | ancrage {pd.Timestamp(anchor_ts, unit='ms'):%Y-%m-%d %H:%M} | {n_paths} trajectoires")

    # 2. Oracle : entraîné sur CETTE chaîne et AVANT l'ancrage, sinon rien
    try:
        oracle = MetaOracle.load(oracle_path, engine, bt_cfg, as_of_ms=anchor_ts)
    except (FileNotFoundError, OracleMismatch) as e:
        logger.warning(f"XGB désactivé, HJB seul : {e}")
        oracle = None

    # 3. Prolongements synthétiques
    sde = GenerativeSDEEngine(model_path=latest_checkpoint())
    horizon = sde.model_seq_len - 1 if horizon is None else int(horizon)
    markets = synthetic_markets(sde, f"{pair_a}x{pair_b}", aligned, anchor, horizon, n_paths, seed)

    # 4. HJB, puis HJB + XGB, sur chaque prolongement
    hist_a = aligned["a"].to_numpy()[anchor - warm:anchor + 1]       # l'ancrage est à l'indice `warm`
    hist_b = aligned["b"].to_numpy()[anchor - warm:anchor + 1]
    runs = {"HJB seul": []}
    if oracle is not None:
        runs["HJB + XGB"] = []
    n_divergent = n_trades = n_kept = 0
    for market in tqdm(markets, desc="Stress-test"):
        syn_a, syn_b = market["SYNTH_A"].to_numpy(dtype=float), market["SYNTH_B"].to_numpy(dtype=float)
        if diverged(syn_a) or diverged(syn_b):
            n_divergent += 1                                          # rejetée et comptée, pas masquée
            continue
        pa = np.concatenate([hist_a, syn_a[1:]])                      # la barre 0 générée EST l'ancrage
        pb = np.concatenate([hist_b, syn_b[1:]])
        signal = engine.generate(np.arange(len(pa)), pa, pb)
        ahead = signal.slice(warm, len(pa))
        runs["HJB seul"].append(bt.run(ahead, lock_hedge=True)["metrics"])
        if oracle is not None:
            res = bt.run(ahead, lock_hedge=True, keep_trade=oracle.gate(np.log(pa), np.log(pb), signal, offset=warm))
            runs["HJB + XGB"].append(res["metrics"])
            n_trades += len(res["trades"])
            n_kept += sum(t["kept"] for t in res["trades"])

    # 5. Bilan
    if n_divergent:
        logger.warning(f"{n_divergent}/{len(markets)} trajectoires divergentes rejetées (le taux doit rester marginal).")
    if not runs["HJB seul"]:
        logger.error("Aucune trajectoire exploitable.")
        return None
    runs = {name: pd.DataFrame(rows) for name, rows in runs.items()}
    table = pd.DataFrame({name: summarize(m) for name, m in runs.items()})
    logger.info(f"{len(runs['HJB seul'])} prolongements de {horizon} barres\n" + table.to_string(float_format=lambda v: f"{v:.3f}"))
    if oracle is not None:
        logger.info(f"XGB (seuil {oracle.threshold:.2f}) : {n_kept}/{n_trades} trades gardés.")
    plot_distributions(runs, f"{pair_a}/{pair_b}", os.path.join(PATHS_DIR, f"{pair_a}x{pair_b}_cvar_plot.png"))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_a", type=str, default="ZEC_USDT")
    parser.add_argument("--pair_b", type=str, default="XRP_USDT")
    parser.add_argument("--n_paths", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=None, help="barres générées (défaut : longueur d'entraînement du générateur)")
    parser.add_argument("--anchor_offset", type=int, default=0, help="ancrage à N barres de la fin de l'historique")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle", type=str, default=ORACLE_PATH)
    args = parser.parse_args()

    run_generative_cvar(args.pair_a, args.pair_b, args.n_paths, args.horizon, args.anchor_offset, args.seed, args.oracle)
