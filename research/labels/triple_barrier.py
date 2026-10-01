"""
ML Labeler pour Pairs Trading
Output : DataFrame avec features @ entry + label basé sur le vrai P&L
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from loguru import logger


@dataclass 
class LabeledTrade:
    entry_idx:    int
    exit_idx:     int
    direction:    int       # +1 long spread, -1 short spread
    exit_reason:  str       # "take_profit" | "stop_loss" | "time_stop"
    pnl_pct:      float     # rendement net en %
    label:        int       # +1 profitable, -1 perdant, 0 neutre


class TripleBarrierLabeler:
    """
    Pour chaque point d'entrée potentiel :
    1. Simule le trade (Triple Barrière sur z-score)
    2. Calcule le vrai P&L sur les prix
    3. Assigne le label ML
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold:  float = 0.3,
        stop_loss_z:     float = 4.0,
        time_factor:     float = 2.0,
        min_hold:        int   = 10,
        fee_rate:        float = 0.0004,
        slippage_bps:    float = 2.0,
        neutral_band:    float = 0.001,  # [-0.1%, +0.1%] = label 0
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.stop_loss_z     = stop_loss_z
        self.time_factor     = time_factor
        self.min_hold        = min_hold
        self.cost_per_tx     = fee_rate + slippage_bps / 10_000
        self.neutral_band    = neutral_band

    def _find_exit(
        self, zscores: np.ndarray, entry_idx: int, direction: int, max_hold: int
    ) -> tuple:
        """Trouve l'exit selon les 3 barrières. Returns (exit_idx, reason)."""
        n = len(zscores)

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            z = zscores[j]
            held = j - entry_idx

            # Take profit
            if held >= self.min_hold:
                if direction == -1 and z <= self.exit_threshold:
                    return j, "take_profit"
                if direction == +1 and z >= -self.exit_threshold:
                    return j, "take_profit"

            # Stop loss
            if direction == -1 and z >= self.stop_loss_z:
                return j, "stop_loss"
            if direction == +1 and z <= -self.stop_loss_z:
                return j, "stop_loss"

        # Time stop
        return min(entry_idx + max_hold, n - 1), "time_stop"

    def _compute_pnl_pct(
        self,
        direction: int,
        beta:      float,
        pa_entry:  float,
        pb_entry:  float,
        pa_exit:   float,
        pb_exit:   float,
    ) -> float:
        """P&L net en % du notional, après 4 transactions de frais."""
        ret_a = (pa_exit - pa_entry) / pa_entry
        ret_b = (pb_exit - pb_entry) / pb_entry

        spread_ret = ret_a - beta * ret_b
        gross_pct  = direction * spread_ret

        # 4 transactions : entrée (2 legs) + sortie (2 legs)
        # Notional total tradé = 2 × (1 + β) × notional, aller-retour
        fee_pct = 2 * (1 + abs(beta)) * self.cost_per_tx

        return gross_pct - fee_pct

    def label_all(
        self,
        zscores:   np.ndarray,
        price_a:   np.ndarray,
        price_b:   np.ndarray,
        betas:     np.ndarray,
        half_life: float,
    ) -> List[LabeledTrade]:
        """
        Parcourt toutes les barres, labellise chaque entrée potentielle.
        Un seul trade à la fois (séquentiel, pas de chevauchement).
        """
        n = len(zscores)
        max_hold = max(int(self.time_factor * half_life), self.min_hold + 5)
        trades = []
        i = 0

        while i < n:
            z = zscores[i]

            if z > self.entry_threshold:
                direction = -1
            elif z < -self.entry_threshold:
                direction = +1
            else:
                i += 1
                continue

            exit_idx, exit_reason = self._find_exit(zscores, i, direction, max_hold)

            pnl_pct = self._compute_pnl_pct(
                direction = direction,
                beta      = betas[i],
                pa_entry  = price_a[i],
                pb_entry  = price_b[i],
                pa_exit   = price_a[exit_idx],
                pb_exit   = price_b[exit_idx],
            )

            # Label
            if pnl_pct > self.neutral_band:
                label = 1
            elif pnl_pct < -self.neutral_band:
                label = -1
            else:
                label = 0

            trades.append(LabeledTrade(
                entry_idx   = i,
                exit_idx    = exit_idx,
                direction   = direction,
                exit_reason = exit_reason,
                pnl_pct     = pnl_pct,
                label       = label,
            ))

            i += 1

        logger.info(
            f"Labeler: {len(trades)} trades | "
            f"+1={sum(1 for t in trades if t.label==1)} "
            f"-1={sum(1 for t in trades if t.label==-1)} "
            f" 0={sum(1 for t in trades if t.label==0)}"
        )
        return trades

    def to_dataframe(
        self,
        trades:             List[LabeledTrade],
        signature_features: np.ndarray = None,
        zscores:            np.ndarray = None,
        betas:              np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Construit le DataFrame ML : une ligne par trade,
        features @ moment de l'entrée + label.
        """
        records = []
        for t in trades:
            row = {
                "entry_idx":   t.entry_idx,
                "exit_idx":    t.exit_idx,
                "direction":   t.direction,
                "exit_reason": t.exit_reason,
                "pnl_pct":     round(t.pnl_pct * 100, 4),
                "label":       t.label,
            }

            # Features @ entry
            if zscores is not None:
                row["zscore_entry"] = zscores[t.entry_idx]
            if betas is not None:
                row["beta_entry"] = betas[t.entry_idx]

            # Signature features @ entry (si dispo)
            if signature_features is not None and t.entry_idx < len(signature_features):
                for k, v in enumerate(signature_features[t.entry_idx]):
                    row[f"sig_{k}"] = v

            records.append(row)

        return pd.DataFrame(records)
