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
    Cost model corrigé et uniformisé :
    - Coût appliqué à CHAQUE transaction (ouverture OU fermeture)
    - Une transaction = 2 legs simultanés (long A + short B)
    - Funding rate sur positions overnight
    """

    def __init__(self, scenario: FeeScenario):
        self.scenario = scenario
        self.fee      = scenario.fee_rate
        self.slip     = scenario.slippage_bps / 10_000
        self.funding  = scenario.funding_rate_8h

    def transaction_cost(self, notional_per_leg: float) -> float:
        """
        Coût d'UNE transaction sur la paire (ouverture OU fermeture).
        2 legs simultanés (Asset A + Asset B).
        Un round-trip complet = 2 appels à cette fonction = 4 legs au total.
        """
        return notional_per_leg * (self.fee + self.slip) * 2

    def funding_cost_per_bar(
        self,
        position: float,
        notional_per_leg: float,
        bar_hours: float = 1.0,
    ) -> float:
        """
        Funding rate crypto : payé toutes les 8h.
        On est long/short simultanément → en théorie les fundings se compensent,
        MAIS en pratique les taux diffèrent → on paye conservativement sur les 2 legs.
        """
        funding_per_bar = self.funding * (bar_hours / 8.0)
        return abs(position) * notional_per_leg * funding_per_bar * 2



# ============================================================
# PnL ENGINE CORRIGÉ
# ============================================================

class RealisticPnLEngine:
    def __init__(self, config: BacktestConfig, cost_model: RealisticCostModel):
        self.cfg   = config
        self.costs = cost_model

    def compute(
        self,
        positions: np.ndarray,
        spreads:   np.ndarray,
        price_a:   np.ndarray,
        price_b:   np.ndarray,
        betas:     np.ndarray,
        bar_hours: float = 1.0,
    ) -> pd.DataFrame:

        n = len(positions)

        # --- Returns bar-à-bar ---
        ret_a    = np.zeros(n)
        ret_b    = np.zeros(n)
        ret_a[1:] = np.diff(price_a) / np.where(price_a[:-1] > 0, price_a[:-1], np.nan)
        ret_b[1:] = np.diff(price_b) / np.where(price_b[:-1] > 0, price_b[:-1], np.nan)
        ret_a    = np.nan_to_num(ret_a)
        ret_b    = np.nan_to_num(ret_b)

        shifted_betas     = np.roll(betas, 1)
        shifted_positions = np.roll(positions, 1)
        shifted_positions[0] = 0.0

        spread_return = ret_a - shifted_betas * ret_b
        spread_return = np.nan_to_num(spread_return, nan=0.0)

        # --- Boucle séquentielle : compounding + coûts détaillés ---
        raw_pnl       = np.zeros(n)
        cost_tx_arr   = np.zeros(n)   # coûts de transaction (fees + slippage)
        cost_fund_arr = np.zeros(n)   # coûts de funding
        notional_arr  = np.zeros(n)
        capital       = np.full(n, self.cfg.initial_capital, dtype=np.float64)

        current_notional = 0.0
        prev_pos         = 0.0

        for i in range(1, n):
            cap_prev = capital[i - 1]
            pos_i    = positions[i]

            transition = (pos_i != prev_pos)

            # --- 1) Coûts de transaction (ouverture / fermeture / flip) ---
            if transition:
                # Fermeture de l'ancienne position
                if prev_pos != 0:
                    cost_tx_arr[i] += self.costs.transaction_cost(current_notional)

                # Ouverture de la nouvelle position
                if pos_i != 0:
                    new_notional = cap_prev * self.cfg.position_size
                    cost_tx_arr[i] += self.costs.transaction_cost(new_notional)
                    current_notional = new_notional
                else:
                    current_notional = 0.0

            # --- 2) Funding cost (payé à chaque bar en position) ---
            if current_notional > 0 and pos_i != 0:
                cost_fund_arr[i] = self.costs.funding_cost_per_bar(
                    pos_i, current_notional, bar_hours
                )

            # --- 3) P&L brut du bar ---
            raw_pnl[i] = shifted_positions[i] * spread_return[i] * current_notional

            # --- 4) Mise à jour capital ---
            total_cost      = cost_tx_arr[i] + cost_fund_arr[i]
            capital[i]      = cap_prev + raw_pnl[i] - total_cost
            notional_arr[i] = current_notional
            prev_pos        = pos_i

        net_pnl = raw_pnl - cost_tx_arr - cost_fund_arr

        return pd.DataFrame({
            "position":   positions,
            "spread":     spreads,
            "ret_a":      ret_a,
            "ret_b":      ret_b,
            "beta_r":     shifted_betas,
            "notional":   notional_arr,
            "raw_pnl":    raw_pnl,
            "cost_tx":    cost_tx_arr,
            "cost_fund":  cost_fund_arr,
            "cost":       cost_tx_arr + cost_fund_arr,  # total, pour compat avec PerformanceMetrics
            "net_pnl":    net_pnl,
            "capital":    capital,
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
            signal.entry_long,
            signal.entry_short,
            signal.exit_signal,
            dynamic_max_hold=168 # Default for stress testing
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
