"""
Backtest Engine - Vectorized
Pairs Trading : long A / short B ou inverse
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Callable, Optional
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
    cooldown_bars:    int   = 6
    time_stop_mult:   Optional[float] = 8.0   # arrêt temporel = mult x demi-vie estimée (min 20 barres) ; None = désactivé. 8 : coût < 1,5 % sur OU simulé, garde-fou résiduel
    # False (défaut, chiffres publiés) : demi-vie estimée sur TOUTE la série, donc avec le futur.
    # True : demi-vie lue à l'entrée de chaque trade (θ calibré sur le passé) ; exige lock_hedge=True.
    time_stop_causal: bool  = False

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
        exit_long:  Optional[np.ndarray] = None,
        exit_short: Optional[np.ndarray] = None,
        allow_flip: bool = False,
    ) -> np.ndarray:
        """
        Machine à états : flat -> long/short sur signal d'entrée, retour à flat sur
        signal de sortie ou durée max. Si `exit_long` / `exit_short` sont fournis
        (sorties dépendantes du côté, ex. seuils HJB asymétriques), ils priment sur
        `exit_signal` pour le côté correspondant. Si `allow_flip`, un signal d'entrée du côté
        opposé retourne directement la position (long <-> short, sans passage à plat).
        """

        n          = len(entry_long)
        positions  = np.zeros(n, dtype=np.float32)
        position   = 0
        entry_bar  = 0
        cooldown   = 0

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

                if allow_flip and ((position == 1 and entry_short[i]) or (position == -1 and entry_long[i])):
                    position  = -position
                    entry_bar = i
                    positions[i] = position
                    continue

                max_hold_reached = holding >= dynamic_max_hold

                if position == 1 and exit_long is not None:
                    exit_now = bool(exit_long[i])
                elif position == -1 and exit_short is not None:
                    exit_now = bool(exit_short[i])
                else:
                    exit_now = bool(exit_signal[i])

                if exit_now or max_hold_reached:
                    if holding < 12:
                        cooldown = self.cfg.cooldown_bars
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
        betas:     np.ndarray,
        allocations: Optional[np.ndarray] = None,
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

        # --- Boucle séquentielle : capital et notional compound ---
        raw_pnl      = np.zeros(n)
        cost_per_bar = np.zeros(n)
        notional_arr = np.zeros(n)
        capital      = np.full(n, self.cfg.initial_capital, dtype=np.float64)

        current_notional = 0.0   # notional par leg, figé à l'entrée du trade courant
        prev_pos         = 0.0

        for i in range(1, n):
            cap_prev = capital[i - 1]
            pos_i    = positions[i]

            # P&L du bar : position et notional portés PENDANT la barre i, c'est-à-dire
            # ceux décidés à la barre i-1 (pas de look-ahead). Calculé AVANT la
            # transition de la barre i, sinon la barre de sortie n'est jamais créditée.
            raw_pnl[i] = shifted_positions[i] * spread_return[i] * current_notional

            transition = (pos_i != prev_pos)

            if transition:
                # 1) On ferme l'ancienne position si elle existait (coût sur le notional figé)
                if prev_pos != 0:
                    cost_per_bar[i] += self.costs.transaction_cost(current_notional)

                # 2) On ouvre la nouvelle position si pos_i != 0 (coût sur le nouveau notional)
                if pos_i != 0:
                    alloc_i = float(allocations[i]) if allocations is not None else float(self.cfg.position_size)
                    new_notional = cap_prev * alloc_i
                    cost_per_bar[i] += self.costs.transaction_cost(new_notional)
                    current_notional = new_notional
                else:
                    current_notional = 0.0

            capital[i]      = cap_prev + raw_pnl[i] - cost_per_bar[i]
            notional_arr[i] = current_notional
            prev_pos        = pos_i

        net_pnl = raw_pnl - cost_per_bar

        return pd.DataFrame({
            "position":  positions,
            "spread":    spreads,
            "ret_a":     ret_a,
            "ret_b":     ret_b,
            "beta_r":    shifted_betas,
            "notional":  notional_arr,
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
        prev       = np.concatenate([[0.0], pos[:-1]])
        # Un trade = une entrée (flat -> position, ou flip long <-> short). Les sorties
        # ne sont pas comptées (l'ancienne version comptait entrées + sorties).
        n_trades   = int(((pos != 0) & (pos != prev)).sum())
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

def causal_max_hold(theta_hat: np.ndarray, mult: float, max_half_life: float = 336.0) -> np.ndarray:
    """
    Durée maximale de détention PAR BARRE : mult x ln 2 / θ, avec θ calibré sur le passé (min 20 barres).
    La demi-vie est plafonnée à `max_half_life` (même repli que `compute_kalman_halflife` sans retour à la moyenne).
    """
    theta = np.asarray(theta_hat, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(2.0) / theta
    hl = np.where(np.isfinite(hl) & (hl > 0), np.minimum(hl, max_half_life), max_half_life)
    return np.maximum(np.floor(hl * mult), 20.0)

class BacktestEngine:

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg        = config or BacktestConfig()
        self.costs      = CostModel(self.cfg.fee_rate, self.cfg.slippage_bps)
        self.pm         = PositionManager(self.cfg)
        self.pnl_engine = PnLEngine(self.cfg, self.costs)

    def run(self, signal, label: str = "", allocations: Optional[np.ndarray] = None,
            lock_hedge: bool = False, log_prices: Optional[tuple] = None,
            stop_sd: Optional[float] = None, keep_trade: Optional[Callable[[dict], bool]] = None) -> dict:
        """
        lock_hedge=True : politique à couverture verrouillée à l'entrée (voir research/institutional/locked_policy.py) ;
        `log_prices=(log_a, log_b)` est alors requis (sinon calculé depuis signal.price_a / price_b).

        keep_trade : filtre de méta-label, appelé UNE fois par trade de la politique avec son dictionnaire
        (entry, exit, side, kind, mae_sd). Un trade rejeté est remplacé par une position nulle sur toute sa durée ;
        les autres trades sont inchangés. Pour rester causal le filtre ne doit lire que `entry` et `side` (et le
        passé) : `exit`, `kind` et `mae_sd` sont connus après coup. Exige lock_hedge=True.
        """

        # label = label or f"{signal.symbol_a}×{signal.symbol_b}"
        # logger.info(f"Running backtest : {label}")

        if keep_trade is not None and not lock_hedge:
            raise ValueError("keep_trade exige lock_hedge=True (le filtre porte sur les trades de la politique verrouillée)")

        if self.cfg.time_stop_mult is None:
            dynamic_max_hold = 10 ** 9
        elif self.cfg.time_stop_causal:
            if not lock_hedge:
                raise ValueError("time_stop_causal exige lock_hedge=True")
            dynamic_max_hold = causal_max_hold(signal.theta_hat, self.cfg.time_stop_mult)
        else:
            hl = compute_kalman_halflife(signal.spreads)
            dynamic_max_hold = max(int(hl * self.cfg.time_stop_mult), 20)

        # print(f'Dynamic Half Life : {hl}')
        betas_used = signal.betas
        trades = None
        if lock_hedge:
            from research.institutional.locked_policy import locked_positions
            if log_prices is None:
                log_prices = (np.log(np.asarray(signal.price_a, dtype=float)), np.log(np.asarray(signal.price_b, dtype=float)))
            positions, betas_used, trades = locked_positions(signal, log_prices[0], log_prices[1], dynamic_max_hold,
                                                              stop_sd=stop_sd, return_trades=True)
            if keep_trade is not None:
                for t in trades:
                    t["kept"] = bool(keep_trade(t))
                    if not t["kept"]:
                        # Trade encore ouvert en fin de série : la dernière barre est aussi en position.
                        end = len(positions) if t["kind"] == "fin de série" else t["exit"]
                        positions[t["entry"]:end] = 0
        else:
            positions = self.pm.compute_positions(
                signal.entry_long,
                signal.entry_short,
                signal.exit_signal,
                dynamic_max_hold,
                exit_long=getattr(signal, "exit_long", None),
                exit_short=getattr(signal, "exit_short", None),
                allow_flip=bool(getattr(signal, "allow_flip", False)),
            )

        df = self.pnl_engine.compute(
            positions,
            signal.spreads,
            signal.price_a,
            signal.price_b,
            betas_used,
            allocations=allocations,
        )

        metrics = PerformanceMetrics.compute(df)

        # logger.info(
        #     f"{label:35s} | "
        #     f"Sharpe={metrics['sharpe']:6.3f} | "
        #     f"Ret={metrics['total_return_pct']:8.2f}% | "
        #     f"MDD={metrics['max_dd']:7.2f}% | "
            # f"Trades={metrics['n_trades']}"
        # )

        return {"metrics": metrics, "df": df, "positions": positions, "trades": trades}
