"""
research/backtest/stress_test.py
Stress test frais + slippage réalistes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from itertools import product

from research.backtest.engine import BacktestEngine, BacktestConfig


@dataclass
class FeeScenario:
    name:           str
    fee_rate:       float   # taker fee (fraction)
    slippage_bps:   float   # slippage en bps
    funding_rate_8h: float  # funding rate crypto (8h), 0 si non applicable
    description:    str = ""

    def total_cost_bps(self) -> float:
        return self.fee_rate * 10_000 + self.slippage_bps

# Scénarios réalistes Binance Perp / Spot
FEE_SCENARIOS = {
    "optimistic": FeeScenario(
        name="optimistic",
        fee_rate=0.0002,        # 2bps maker fee (VIP tier)
        slippage_bps=0.5,       # très liquide, spread tight
        funding_rate_8h=0.0,
        description="Maker VIP, top liquidity",
    ),
    "base": FeeScenario(
        name="base",
        fee_rate=0.0004,        # 4bps taker standard
        slippage_bps=2.0,       # 2bps slippage réaliste
        funding_rate_8h=0.0001, # 0.01% / 8h = ~10% annuel
        description="Taker standard Binance",
    ),
    "realistic": FeeScenario(
        name="realistic",
        fee_rate=0.0006,        # 6bps (taker + exchange fee)
        slippage_bps=5.0,       # 5bps slippage marché chargé
        funding_rate_8h=0.0003, # 0.03% / 8h = ~33% annuel
        description="Conditions réalistes peak hours",
    ),
    "stressed": FeeScenario(
        name="stressed",
        fee_rate=0.001,         # 10bps (stress / altcoins illiquides)
        slippage_bps=15.0,      # 15bps slippage volatile
        funding_rate_8h=0.001,  # 0.1% / 8h = ~110% annuel (flash crash)
        description="Stress: illiquidité + funding extrême",
    ),
}


class RealisticCostModel:
    """
    Cost model corrigé :
    - 4 transactions par round-trip (open A, open B, close A, close B)
    - Funding rate sur positions overnight
    - Slippage volatility-adjusted (optionnel)
    """

    def __init__(self, scenario: FeeScenario):
        self.scenario = scenario
        self.fee      = scenario.fee_rate
        self.slip     = scenario.slippage_bps / 10_000
        self.funding  = scenario.funding_rate_8h

    def round_trip_cost(self, notional_per_leg: float) -> float:
        """
        Coût complet d'un aller-retour sur une paire.
        4 transactions : open_A, open_B, close_A, close_B
        """
        cost_per_transaction = notional_per_leg * (self.fee + self.slip)
        return cost_per_transaction * 4  # 4 legs

    def funding_cost_per_bar(
        self,
        position: float,        # +1 / -1
        notional: float,
        bar_hours: float = 1.0,
    ) -> float:
        """
        Funding rate crypto : payé toutes les 8h.
        Short position reçoit le funding, long le paie (en général).
        On est toujours long/short simultanément → funding net ≈ 0
        MAIS si funding très élevé, l'un des legs paye plus.
        Simplification conservative : on paie toujours.
        """
        funding_per_bar = self.funding * (bar_hours / 8.0)
        # On multiplie par 2 pour couvrir grossièrement le notionnel sur les 2 legs
        return abs(position) * notional * funding_per_bar * 2


# ============================================================
# PnL ENGINE CORRIGÉ
# ============================================================

class RealisticPnLEngine:
    """
    Corrige les bugs du PnLEngine original :
    1. Beta-weighted spread return
    2. 4 legs de coût
    3. Funding rate
    """

    def __init__(self, config: BacktestConfig, cost_model: RealisticCostModel):
        self.cfg   = config
        self.costs = cost_model

    def compute(
        self,
        positions:  np.ndarray,
        spreads:    np.ndarray,
        price_a:    np.ndarray,
        price_b:    np.ndarray,
        betas:      np.ndarray,
        bar_hours:  float = 1.0,
    ) -> pd.DataFrame:

        n = len(positions)

        # --- Returns bar-à-bar ---
        ret_a = np.zeros(n)
        ret_b = np.zeros(n)
        ret_a[1:] = np.diff(price_a) / np.where(price_a[:-1] > 0, price_a[:-1], 1e-9)
        ret_b[1:] = np.diff(price_b) / np.where(price_b[:-1] > 0, price_b[:-1], 1e-9)
        ret_a = np.nan_to_num(ret_a, nan=0.0, posinf=0.0, neginf=0.0)
        ret_b = np.nan_to_num(ret_b, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ FIX 1 : Beta-weighted spread return
        # Long A, Short beta*B → P&L = ret_a - beta * ret_b
        beta_adjusted_return = ret_a - betas * ret_b

        notional_per_leg = self.cfg.initial_capital * self.cfg.position_size

        # Raw P&L (Fix look-ahead bias)
        shifted_positions = np.roll(positions, 1)
        shifted_positions[0] = 0.0
        raw_pnl = shifted_positions * beta_adjusted_return * notional_per_leg

        # ✅ FIX 2 : 4 legs sur round-trip
        pos_change  = np.diff(positions, prepend=0.0)
        is_entry    = (positions != 0) & (pos_change != 0)   # ouverture
        is_exit     = (positions == 0) & (pos_change != 0)   # fermeture

        cost_per_bar = np.zeros(n)

        # Coût à l'entrée
        cost_per_bar[is_entry] = self.costs.round_trip_cost(notional_per_leg)

        # ✅ FIX 3 : Funding rate (payé à chaque bar en position)
        funding_per_bar = np.array([
            self.costs.funding_cost_per_bar(positions[i], notional_per_leg, bar_hours)
            for i in range(n)
        ])

        total_cost = cost_per_bar + funding_per_bar
        net_pnl    = raw_pnl - total_cost

        capital = self.cfg.initial_capital + np.cumsum(net_pnl)

        return pd.DataFrame({
            "position":    positions,
            "spread":      spreads,
            "ret_a":       ret_a,
            "ret_b":       ret_b,
            "beta":        betas,
            "raw_pnl":     raw_pnl,
            "cost_tx":     cost_per_bar,
            "cost_fund":   funding_per_bar,
            "total_cost":  total_cost,
            "net_pnl":     net_pnl,
            "capital":     capital,
        })


# ============================================================
# STRESS TEST ENGINE
# ============================================================

class StressTestEngine:

    def __init__(self, backtest_cfg: Optional[BacktestConfig] = None):
        self.bt_cfg = backtest_cfg or BacktestConfig()

    def run_scenario(
        self,
        signal,
        scenario: FeeScenario,
        label: str = "",
    ) -> dict:
        """Run un seul scénario de frais sur un signal."""
        from research.backtest.engine import PositionManager
        from research.backtest.engine import PerformanceMetrics

        cost_model = RealisticCostModel(scenario)
        pnl_engine = RealisticPnLEngine(self.bt_cfg, cost_model)
        pm         = PositionManager(self.bt_cfg)

        positions = pm.compute_positions(
            signal.zscores,
            signal.entry_long,
            signal.entry_short,
        )

        df = pnl_engine.compute(
            positions,
            signal.spreads,
            signal.price_a,
            signal.price_b,
            signal.betas,
        )

        metrics = PerformanceMetrics.compute(df)

        # Coûts totaux en % du capital initial
        total_fees    = df["cost_tx"].sum()
        total_funding = df["cost_fund"].sum()
        metrics["total_fees_pct"]    = round(total_fees / self.bt_cfg.initial_capital * 100, 2)
        metrics["total_funding_pct"] = round(total_funding / self.bt_cfg.initial_capital * 100, 2)
        metrics["cost_drag_pct"]     = round((total_fees + total_funding) / self.bt_cfg.initial_capital * 100, 2)
        metrics["scenario"]          = scenario.name

        return {"metrics": metrics, "df": df}

    def run_all_scenarios(self, signal, label: str = "") -> pd.DataFrame:
        """Run les 4 scénarios sur un signal, retourne DataFrame comparatif."""
        rows = []
        for name, scenario in FEE_SCENARIOS.items():
            result  = self.run_scenario(signal, scenario, label)
            metrics = result["metrics"]
            metrics["pair"] = label
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("scenario")
        return df

    def run_pair_grid(
        self,
        signals: dict,          # {pair_label: PairSignal}
    ) -> pd.DataFrame:
        """
        Stress test sur toutes les paires × tous les scénarios.
        Returns: MultiIndex DataFrame (pair, scenario)
        """
        all_rows = []
        for label, signal in signals.items():
            logger.info(f"Stress testing {label}...")
            df_scenarios = self.run_all_scenarios(signal, label)
            df_scenarios["pair"] = label
            all_rows.append(df_scenarios)

        result = pd.concat(all_rows)
        result = result.reset_index().set_index(["pair", "scenario"])
        return result


# ============================================================
# ANALYSE DES RÉSULTATS
# ============================================================

def analyze_stress_results(stress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table : paires en lignes, métriques clés par scénario en colonnes.
    Focus sur sharpe + total_return + cost_drag.
    """
    metrics_of_interest = ["sharpe", "total_return_pct", "cost_drag_pct", "max_dd", "calmar"]

    rows = []
    pairs = stress_df.index.get_level_values("pair").unique()

    for pair in pairs:
        row = {"pair": pair}
        for scenario in FEE_SCENARIOS.keys():
            try:
                data = stress_df.loc[(pair, scenario)]
                for m in metrics_of_interest:
                    row[f"{scenario}_{m}"] = data[m]
            except KeyError:
                pass

        # Robustesse : sharpe stressed / sharpe optimistic
        if f"optimistic_sharpe" in row and f"stressed_sharpe" in row:
            opt = row["optimistic_sharpe"]
            row["stress_retention"] = round(row["stressed_sharpe"] / opt, 3) if opt > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("pair").sort_values("realistic_sharpe", ascending=False)


def print_stress_report(stress_df: pd.DataFrame, pair: str):
    """Rapport détaillé pour une paire."""
    print(f"\n{'='*60}")
    print(f"STRESS TEST REPORT : {pair}")
    print(f"{'='*60}")

    cols = ["sharpe", "total_return_pct", "max_dd", "cost_drag_pct",
            "total_fees_pct", "total_funding_pct", "n_trades", "calmar"]

    try:
        sub = stress_df.loc[pair][cols]
        print(sub.to_string())
    except KeyError:
        print(f"Pair {pair} not found")
