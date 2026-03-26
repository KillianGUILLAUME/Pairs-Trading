"""
Backtest Engine - Vectorized
Pairs Trading : long A / short B ou inverse
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Optional
from loguru import logger

# ============================================================
# CONFIG
# ============================================================

@dataclass
class BacktestConfig:
    initial_capital:  float = 100_000.0
    position_size:    float = 0.10
    entry_threshold:  float = 2.0
    exit_threshold:   float = 0.3
    stop_loss_z:      float = 4.0
    max_holding_bars: int   = 336
    fee_rate:         float = 0.0004
    slippage_bps:     float = 1.0
    beta_window:      int   = 120   # rolling OLS sur returns

# ============================================================
# COST MODEL
# ============================================================

class CostModel:
    def __init__(self, fee_rate: float, slippage_bps: float):
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10_000

    def transaction_cost(self, notional: float) -> float:
        # Appelé à chaque changement de position (entrée ET sortie), donc on compte 2 legs (demi aller-retour)
        return notional * (self.fee_rate + self.slippage) * 2

# ============================================================
# POSITION MANAGER
# ============================================================

class PositionManager:
    def __init__(self, config: BacktestConfig):
        self.cfg = config

    def compute_positions(
        self,
        entry_long:  np.ndarray,
        entry_short: np.ndarray,
        exit_signal: np.ndarray,
        dynamic_max_hold: int,
    ) -> np.ndarray:

        n          = len(entry_long)
        positions  = np.zeros(n, dtype=np.float32)
        position   = 0
        entry_bar  = 0
        cooldown   = 0
        COOLDOWN_BARS = 0

        for i in range(1, n):

            if cooldown > 0:
                cooldown -= 1
                positions[i] = 0
                continue

            if position == 0:
                if entry_long[i]:
                    position  = 1
                    entry_bar = i
                elif entry_short[i]:
                    position  = -1
                    entry_bar = i
            else:
                holding  = i - entry_bar

                max_hold_reached = holding >= dynamic_max_hold

                if exit_signal[i] or max_hold_reached:
                    if holding < 12:
                        cooldown = COOLDOWN_BARS
                    position = 0

            positions[i] = position

        return positions

# ============================================================
# P&L ENGINE
# ============================================================

class PnLEngine:
    def __init__(self, config: BacktestConfig, cost_model: CostModel):
        self.cfg   = config
        self.costs = cost_model

    def compute(
        self,
        positions: np.ndarray,
        spreads:   np.ndarray,
        price_a:   np.ndarray,
        price_b:   np.ndarray,
        betas:     np.ndarray,   # ignoré — recalculé en espace returns
    ) -> pd.DataFrame:

        n = len(positions)

        # --- Returns bar-à-bar ---
        ret_a    = np.zeros(n)
        ret_b    = np.zeros(n)
        ret_a[1:] = np.diff(price_a) / np.where(price_a[:-1] > 0, price_a[:-1], np.nan)
        ret_b[1:] = np.diff(price_b) / np.where(price_b[:-1] > 0, price_b[:-1], np.nan)
        ret_a    = np.nan_to_num(ret_a)
        ret_b    = np.nan_to_num(ret_b)

        shifted_betas = np.roll(betas, 1)

        # --- Spread return beta-hedged (Avec le Kalman Beta) ---
        spread_return = ret_a - shifted_betas * ret_b

        # --- Notional fixe sur le leg A ---
        notional = self.cfg.initial_capital * self.cfg.position_size

        shifted_positions = np.roll(positions, 1)
        shifted_positions[0] = 0.0

        raw_pnl = shifted_positions * spread_return * notional
        raw_pnl = np.nan_to_num(raw_pnl, nan=0.0)

        # --- Coûts aux transitions uniquement ---
        pos_change   = np.diff(positions, prepend=0.0)
        cost_per_bar = np.zeros(n)
        cost_per_bar[pos_change != 0] = self.costs.transaction_cost(notional)

        net_pnl = raw_pnl - cost_per_bar
        capital = self.cfg.initial_capital + np.cumsum(net_pnl)

        return pd.DataFrame({
            "position":  positions,
            "spread":    spreads,
            "ret_a":     ret_a,
            "ret_b":     ret_b,
            "beta_r":    shifted_betas, #kalman beta
            "raw_pnl":   raw_pnl,
            "cost":      cost_per_bar,
            "net_pnl":   net_pnl,
            "capital":   capital,
        })

# ============================================================
# PERFORMANCE METRICS
# ============================================================

class PerformanceMetrics:

    @staticmethod
    def compute(df: pd.DataFrame, freq_hours: float = 1.0) -> dict:
        bars_per_year = int(8760 / freq_hours)

        equity = df["capital"].values
        pnl    = df["net_pnl"].values

        ret    = np.diff(equity) / equity[:-1]
        ret    = np.nan_to_num(ret)

        mu     = ret.mean()
        sigma  = ret.std()
        sharpe = (mu / sigma * np.sqrt(bars_per_year)) if sigma > 0 else 0.0

        neg      = ret[ret < 0]
        downside = neg.std() if len(neg) > 0 else 0.0
        sortino  = (mu / downside * np.sqrt(bars_per_year)) if downside > 0 else 0.0

        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / np.where(peak > 0, peak, 1)
        max_dd = dd.min()

        total_ret = (equity[-1] / equity[0] - 1)
        n_years   = len(pnl) / bars_per_year
        ann_ret   = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        calmar    = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        pos        = df["position"].values
        pos_change = np.diff(pos, prepend=0.0)
        n_trades   = int((pos_change != 0).sum())
        pos_bars   = (pos != 0).sum()

        win_pnl  = pnl[pnl > 0].sum()
        loss_pnl = abs(pnl[pnl < 0].sum())
        pf       = win_pnl / loss_pnl if loss_pnl > 0 else np.nan

        return {
            "total_return_pct":  round(total_ret * 100, 2),
            "annual_return_pct": round(ann_ret * 100, 2),
            "sharpe":            round(sharpe, 3),
            "sortino":           round(sortino, 3),
            "max_dd":            round(max_dd * 100, 2),
            "calmar":            round(calmar, 3),
            "n_trades":          n_trades,
            "pct_in_market":     round(pos_bars / len(pos) * 100, 1),
            "profit_factor":     round(pf, 3),
            "final_capital":     round(equity[-1], 2),
        }

# ============================================================
# BACKTEST ENGINE
# ============================================================

def compute_kalman_halflife(spread_array: np.ndarray) -> float:
    """
    Calcule le Half-Life mathématique d'une série temporelle
    via la régression d'Ornstein-Uhlenbeck.
    """
    y = spread_array[~np.isnan(spread_array)]
    
    if len(y) < 100:
        return 72.0 
        
    y_lag = y[:-1]
    dy = y[1:] - y_lag
    
    X = sm.add_constant(y_lag)
    
    res = sm.OLS(dy, X).fit()
    lam = res.params[1] 
    
    if lam >= 0:
        return 336.0 
        
    hl = -np.log(2) / lam
    return float(hl)

class BacktestEngine:

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg        = config or BacktestConfig()
        self.costs      = CostModel(self.cfg.fee_rate, self.cfg.slippage_bps)
        self.pm         = PositionManager(self.cfg)
        self.pnl_engine = PnLEngine(self.cfg, self.costs)

    def run(self, signal, label: str = "") -> dict:

        label = label or f"{signal.symbol_a}×{signal.symbol_b}"
        # logger.info(f"Running backtest : {label}")

        hl = compute_kalman_halflife(signal.spreads)
        time_factor = 2.0
        dynamic_max_hold = max(int(hl * time_factor), 20)

        print(f'Dynamic Half Life : {hl}')
        positions = self.pm.compute_positions(
            signal.entry_long,
            signal.entry_short,
            signal.exit_signal,
            dynamic_max_hold,
        )

        df = self.pnl_engine.compute(
            positions,
            signal.spreads,
            signal.price_a,
            signal.price_b,
            signal.betas,
        )

        metrics = PerformanceMetrics.compute(df)

        logger.info(
            f"{label:35s} | "
            f"Sharpe={metrics['sharpe']:6.3f} | "
            f"Ret={metrics['total_return_pct']:8.2f}% | "
            f"MDD={metrics['max_dd']:7.2f}% | "
            f"Trades={metrics['n_trades']}"
        )

        return {"metrics": metrics, "df": df, "positions": positions}
