"""
research/optimization/optimize_signals.py

Optimisation Bayésienne des hyperparamètres de signal avec Optuna.
Utilise Walk-Forward Optimization (WFO) pour éviter l'overfitting.

Pipeline :
    1. Découpe l'historique en K blocs train/test
    2. Pour chaque trial Optuna, évalue sur tous les folds WFO
    3. L'objectif est le Sharpe médian out-of-sample (robuste aux outliers)

Usage:
    python -m research.optimization.optimize_signals                 # Défaut ADA/AVAX
    python -m research.optimization.optimize_signals --n_trials 300  # Plus de budget
    python -m research.optimization.optimize_signals --pair_a ZEC_USDT --pair_b XRP_USDT
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import optuna
from loguru import logger
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.signals.signal_generator import SignalGenerator
from research.backtest.engine import (
    BacktestConfig, BacktestEngine, PerformanceMetrics,
    PositionManager, PnLEngine, CostModel, compute_kalman_halflife,
)
from research.backtest.stress_test import FEE_SCENARIOS, RealisticCostModel, RealisticPnLEngine


# ============================================================
#  1. Walk-Forward Split
# ============================================================

@dataclass
class WFOConfig:
    """Configuration du Walk-Forward."""
    n_folds: int = 5
    min_train_bars: int = 2000     # Minimum de barres d'entraînement
    min_test_bars: int = 500       # Minimum de barres de test


def walk_forward_splits(
    n_total: int,
    config: WFOConfig = None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Walk-Forward avec blocs de test STRICTEMENT non-chevauchants.
    
    Schéma (expanding window) :
        |=== Train 1 ===|-- Test 1 --|                          
        |====== Train 2 ======|-- Test 2 --|                    
        |========= Train 3 =========|-- Test 3 --|              
        |============ Train 4 ============|-- Test 4 --|        
        |=============== Train 5 ===============|-- Test 5 --|  
    
    Chaque test block est indépendant. Zéro overlap.
    Le train s'étend à chaque fold (expanding window).
    
    Returns: list of ((train_start, train_end), (test_start, test_end))
    """
    if config is None:
        config = WFOConfig()
    
    # On réserve min_train_bars au début, le reste est divisé en n_folds blocs test
    available_for_test = n_total - config.min_train_bars
    
    if available_for_test < config.min_test_bars * config.n_folds:
        # Pas assez de données → réduire le nombre de folds
        actual_folds = max(1, available_for_test // config.min_test_bars)
    else:
        actual_folds = config.n_folds
    
    test_block_size = available_for_test // actual_folds
    
    if test_block_size < config.min_test_bars:
        test_block_size = config.min_test_bars
        actual_folds = available_for_test // test_block_size
    
    splits = []
    
    for fold in range(actual_folds):
        # Test block : contiguous, non-overlapping
        test_start = config.min_train_bars + fold * test_block_size
        test_end = test_start + test_block_size
        
        if test_end > n_total:
            test_end = n_total
            if test_end - test_start < config.min_test_bars:
                break
        
        # Train : tout ce qui est AVANT le test block (expanding window)
        train_start = 0
        train_end = test_start
        
        if train_end - train_start < config.min_train_bars:
            break
        
        splits.append(((train_start, train_end), (test_start, test_end)))
    
    return splits


# ============================================================
#  2. Objective Function
# ============================================================

def evaluate_params_on_fold(
    price_a: np.ndarray,
    price_b: np.ndarray,
    timestamps: np.ndarray,
    train_idx: tuple[int, int],
    test_idx: tuple[int, int],
    signal_params: dict,
    backtest_params: dict,
    fee_scenario: str = "base",
) -> dict:
    """
    Évalue un jeu de paramètres sur un fold WFO.
    
    1. Calibre le Kalman sur le TRAIN
    2. Génère les signaux sur TRAIN+TEST (le Kalman a besoin de warmup)
    3. Évalue les métriques uniquement sur TEST
    """
    train_s, train_e = train_idx
    test_s, test_e = test_idx
    
    # On démarre le signal depuis le début du train pour le warmup Kalman
    full_s = train_s
    full_e = test_e
    
    pa = price_a[full_s:full_e]
    pb = price_b[full_s:full_e]
    ts = timestamps[full_s:full_e]
    
    if len(pa) < 500:
        return {"sharpe": 0.0, "total_return_pct": 0.0, "max_dd": 0.0, "n_trades": 0}
    
    try:
        sig_gen = SignalGenerator(
            zscore_window=signal_params["zscore_window"],
            entry_threshold=signal_params["entry_threshold"],
            exit_threshold=signal_params["exit_threshold"],
            compute_signature=False,
        )
        
        signal = sig_gen.generate(
            ts, pa, pb,
            symbol_a="A", symbol_b="B",
            auto_calibrate=True,
            use_regime_filter=False,
        )
        
        # Slice TEST uniquement (offset dans le signal)
        test_offset = test_s - full_s
        test_len = test_e - test_s
        signal_test = signal.slice(test_offset, test_offset + test_len)
        
        # Backtest config
        bt_config = BacktestConfig(
            initial_capital=backtest_params["initial_capital"],
            position_size=backtest_params["position_size"],
            entry_threshold=signal_params["entry_threshold"],
            exit_threshold=signal_params["exit_threshold"],
            stop_loss_z=backtest_params["stop_loss_z"],
            max_holding_bars=backtest_params["max_holding_bars"],
            cooldown_bars=backtest_params["cooldown_bars"],
        )
        
        # PnL avec coûts réalistes
        fee_scenario = FEE_SCENARIOS[fee_scenario]
        cost_model = RealisticCostModel(fee_scenario)
        pnl_engine = RealisticPnLEngine(bt_config, cost_model)
        pm = PositionManager(bt_config)
        
        hl = compute_kalman_halflife(signal_test.spreads)
        dynamic_max_hold = max(int(hl * 2.0), 20)
        
        positions = pm.compute_positions(
            signal_test.entry_long,
            signal_test.entry_short,
            signal_test.exit_signal,
            dynamic_max_hold,
        )
        
        df = pnl_engine.compute(
            positions, signal_test.spreads,
            signal_test.price_a, signal_test.price_b,
            signal_test.betas,
        )
        
        # Sanitize NaN
        df["net_pnl"] = df["net_pnl"].fillna(0.0)
        df["raw_pnl"] = df["raw_pnl"].fillna(0.0)
        df["cost_tx"] = df["cost_tx"].fillna(0.0)
        df["cost_fund"] = df["cost_fund"].fillna(0.0)
        # df["capital"] = bt_config.initial_capital + df["net_pnl"].cumsum() already done in compute
        
        metrics = PerformanceMetrics.compute(df)

        # Calcul du cost drag
        gross_pnl = df["raw_pnl"].sum()  # PnL brut avant frais
        total_costs = df["cost_tx"].sum() + df["cost_fund"].sum()

        if gross_pnl > 0:
            cost_drag = total_costs / gross_pnl
        else:
            cost_drag = np.inf  # Pas d'edge du tout

        metrics["cost_drag"] = cost_drag
        metrics["gross_pnl"] = gross_pnl
        metrics["total_costs"] = total_costs

        return metrics
        
    except Exception as e:
        _default = {
            "sharpe": 0.0, "total_return_pct": 0.0,
            "max_dd": 0.0, "n_trades": 0,
            "cost_drag": np.inf, "gross_pnl": 0.0, "total_costs": 0.0
        }
        logger.warning(f"evaluate_params_on_fold failed: {e}")
        return {**_default, "error": str(e)}



def create_objective(
    price_a: np.ndarray,
    price_b: np.ndarray,
    timestamps: np.ndarray,
    wfo_config: WFOConfig = None,
    fee_scenario: str = "base",
):
    if wfo_config is None:
        wfo_config = WFOConfig()

    splits = walk_forward_splits(len(price_a), wfo_config)
    logger.info(f"📐 WFO configuré : {len(splits)} folds")
    for i, ((ts, te), (vs, ve)) in enumerate(splits):
        logger.info(f"   Fold {i}: train [{ts}:{te}] ({te-ts} bars) → test [{vs}:{ve}] ({ve-vs} bars)")

    def objective(trial: optuna.Trial) -> float:
        # ── Search Space ──
        signal_params = {
            "entry_threshold": trial.suggest_float("entry_threshold", 1.5, 3.5, step=0.1),
            "exit_threshold":  trial.suggest_float("exit_threshold",  0.1,  1.0, step=0.05),
            "zscore_window":   trial.suggest_int(  "zscore_window",   48,   336, step=12),
        }
        backtest_params = {
            "initial_capital":  100_000,
            "position_size":    trial.suggest_float("position_size",    0.05, 0.25, step=0.01),
            "stop_loss_z":      trial.suggest_float("stop_loss_z",      3.0,  6.0,  step=0.5),
            "max_holding_bars": trial.suggest_int(  "max_holding_bars", 72,   672,  step=24),
            "cooldown_bars":    trial.suggest_int(  "cooldown_bars",    2,    24,   step=2),
        }

        # ── Évaluation sur chaque fold ──
        fold_metrics = []

        for (train_start, train_end), (test_start, test_end) in splits:  # ✅ unpacking correct
            metrics = evaluate_params_on_fold(
                price_a, price_b, timestamps,
                train_idx=(train_start, train_end),
                test_idx=(test_start, test_end),
                signal_params=signal_params,
                backtest_params=backtest_params,
                fee_scenario=fee_scenario,
            )
            fold_metrics.append(metrics)

        # ── Sanitize helper ──
        def _clean(x):
            return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

        # ── Extraction ──
        fold_sharpes = [_clean(m.get("sharpe"))           for m in fold_metrics]
        fold_returns = [_clean(m.get("total_return_pct")) for m in fold_metrics]
        fold_dds     = [_clean(m.get("max_dd"))           for m in fold_metrics]
        fold_trades  = [m.get("n_trades", 0)              for m in fold_metrics]

        cost_drags = [
            m["cost_drag"] for m in fold_metrics
            if m.get("cost_drag") is not None and np.isfinite(m["cost_drag"])
        ]

        # ── Agrégation ──
        median_sharpe    = float(np.median(fold_sharpes))
        worst_dd         = float(np.min(fold_dds))
        avg_trades       = float(np.mean(fold_trades))
        median_cost_drag = float(np.median(cost_drags)) if cost_drags else np.inf

        # ── Pénalités continues (pas de rejet dur) ──
        dd_penalty    = 0.5 * abs(worst_dd) / 100.0          # normalisé en fraction
        trade_penalty = max(0.0, (5 - avg_trades)) * 0.2     # kick si < 5 trades/fold
        cost_penalty  = max(0.0, median_cost_drag - 0.30) * 2.0  # kick après 30% drag

        score = median_sharpe - dd_penalty - trade_penalty - cost_penalty

        # ── User attrs (toujours loggés avant le return) ──
        trial.set_user_attr("median_sharpe",    round(median_sharpe, 3))
        trial.set_user_attr("worst_dd",         round(worst_dd, 2))
        trial.set_user_attr("avg_trades",       round(avg_trades, 1))
        trial.set_user_attr("median_cost_drag", round(median_cost_drag, 3))
        trial.set_user_attr("cost_penalty",     round(cost_penalty, 4))
        trial.set_user_attr("fold_sharpes",     [round(s, 3) for s in fold_sharpes])
        trial.set_user_attr("median_return_pct", round(float(np.median(fold_returns)), 2))
        total_gross  = float(np.nansum([m.get("gross_pnl",   0.0) for m in fold_metrics]))
        total_costs_ = float(np.nansum([m.get("total_costs", 0.0) for m in fold_metrics]))

        trial.set_user_attr("gross_pnl",   round(total_gross,   2))
        trial.set_user_attr("total_costs", round(total_costs_,  2))

        return score

    return objective



# ============================================================
#  3. Runner
# ============================================================

def run_optimization(
    pair_a: str = "ADA_USDT",
    pair_b: str = "AVAX_USDT",
    timeframe: str = "1h",
    n_trials: int = 150,
    n_folds: int = 5,
    fee_scenario: str = "base",
    data_dir: str = None,
    seed: int = 42,
) -> optuna.Study:
    """
    Lance l'optimisation Optuna avec Walk-Forward.
    
    Args:
        n_trials: Nombre d'évaluations (150 ~ 10min sur 8 cores)
        n_folds: Nombre de folds WFO
        fee_scenario: Scénario de frais pour le backtest
    
    Returns: optuna.Study avec les résultats
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "storage", "parquet")
    
    # Chargement des données
    path_a = os.path.join(data_dir, timeframe, f"{pair_a}.parquet")
    path_b = os.path.join(data_dir, timeframe, f"{pair_b}.parquet")
    
    df_a = pd.read_parquet(path_a)[["timestamp", "close"]].rename(columns={"close": "close_a"})
    df_b = pd.read_parquet(path_b)[["timestamp", "close"]].rename(columns={"close": "close_b"})
    df = pd.merge(df_a, df_b, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    
    price_a = df["close_a"].to_numpy()
    price_b = df["close_b"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    
    logger.info(f"{'='*60}")
    logger.info(f"🔬 OPTUNA OPTIMIZER : {pair_a} × {pair_b}")
    logger.info(f"   {len(price_a)} barres | {n_trials} trials | {n_folds} folds WFO")
    logger.info(f"   Fee scenario: {fee_scenario}")
    logger.info(f"{'='*60}")
    
    # Configuration WFO
    wfo = WFOConfig(n_folds=n_folds)
    
    # Création de l'objectif
    objective = create_objective(
        price_a, price_b, timestamps,
        wfo_config=wfo,
        fee_scenario=fee_scenario,
    )
    
    # Sampler TPE (Tree-structured Parzen Estimator)
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=20,    # 20 trials random avant TPE
        multivariate=True,      # Modélise les corrélations entre params
    )
    
    # Étude Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"pairs_{pair_a}x{pair_b}_{fee_scenario}",
    )
    
    # Suppression du logging verbeux Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Callback pour afficher la progression
    def progress_callback(study, trial):
        if trial.number % 10 == 0 or trial.number == n_trials - 1:
            best = study.best_trial
            logger.info(
                f"   Trial {trial.number:3d}/{n_trials} | "
                f"score={trial.value:.3f} | "
                f"best={best.value:.3f} (trial {best.number}) | "
                f"sharpe={best.user_attrs.get('median_sharpe', '?')} | "
                f"dd={best.user_attrs.get('worst_dd', '?')}% | "
                f"Gross PnL: ${best.user_attrs.get('gross_pnl', '?')} | Costs: ${best.user_attrs.get('total_costs', '?')} | Median Drag: {best.user_attrs.get('median_cost_drag', '?')}%"
            )
    
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
    
    # ── Rapport final ──
    best = study.best_trial
    
    print(f"\n{'='*65}")
    print(f"🏆 MEILLEURS PARAMÈTRES — {pair_a} × {pair_b}")
    print(f"{'='*65}")
    print(f"   Score composite        : {best.value:.4f}")
    print(f"   Sharpe médian (OOS)    : {best.user_attrs['median_sharpe']}")
    print(f"   Return médian (OOS)    : {best.user_attrs['median_return_pct']}%")
    print(f"   Pire drawdown          : {best.user_attrs['worst_dd']}%")
    print(f"   Trades moyens/fold     : {best.user_attrs['avg_trades']}")
    print(f"   Sharpes par fold       : {best.user_attrs['fold_sharpes']}")
    print(f"")
    print(f"   ── SIGNAL ──")
    print(f"   entry_threshold        : {best.params['entry_threshold']}")
    print(f"   exit_threshold         : {best.params['exit_threshold']}")
    print(f"   zscore_window          : {best.params['zscore_window']}")
    print(f"")
    print(f"   ── BACKTEST ──")
    print(f"   position_size          : {best.params['position_size']}")
    print(f"   stop_loss_z            : {best.params['stop_loss_z']}")
    print(f"   max_holding_bars       : {best.params['max_holding_bars']}")
    print(f"   cooldown_bars          : {best.params['cooldown_bars']}")
    print(f"{'='*65}")
    
    # Sauvegarde
    out_dir = os.path.join(PROJECT_ROOT, "data", "optimization_results")
    os.makedirs(out_dir, exist_ok=True)
    
    # Top 10 trials
    top = study.trials_dataframe().sort_values("value", ascending=False).head(10)
    top_path = os.path.join(out_dir, f"top10_{pair_a}x{pair_b}_{fee_scenario}.csv")
    top.to_csv(top_path, index=False)
    
    # Best params as YAML-friendly dict
    best_path = os.path.join(out_dir, f"best_{pair_a}x{pair_b}_{fee_scenario}.txt")
    with open(best_path, "w") as f:
        f.write(f"# Optimal params — {pair_a} × {pair_b} ({fee_scenario})\n")
        f.write(f"# Score: {best.value:.4f} | Sharpe: {best.user_attrs['median_sharpe']}\n\n")
        for k, v in best.params.items():
            f.write(f"{k}: {v}\n")
    
    logger.info(f"\n💾 Top 10 : {top_path}")
    logger.info(f"💾 Best   : {best_path}")
    
    return study


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimisation Bayésienne des signaux de trading")
    parser.add_argument("--pair_a", type=str, default="ADA_USDT")
    parser.add_argument("--pair_b", type=str, default="AVAX_USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--n_trials", type=int, default=150)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fees", type=str, default="base",
                        choices=["optimistic", "base", "realistic", "stressed"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    study = run_optimization(
        pair_a=args.pair_a,
        pair_b=args.pair_b,
        timeframe=args.timeframe,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        fee_scenario=args.fees,
        seed=args.seed,
    )
