"""
research/institutional/locked_policy.py

Politique à COUVERTURE VERROUILLÉE À L'ENTRÉE.

Défaut corrigé : le spread-signal est recalculé à chaque rafraîchissement de la couverture
(beta, alpha). Une position ouverte avec (beta_e, alpha_e) est encaissée avec beta_e, mais ses
sorties étaient évaluées sur un spread défini avec le beta COURANT : à chaque rafraîchissement le
signal « revient vers zéro » (le nouvel ajustement absorbe l'écart récent) sans qu'aucun gain
réel n'existe. Mesuré sur paires réelles : écart signal / gain de -262 pb par trade à rafraîchissement
toutes les 24 barres, -49 pb à 2400 barres.

Ici, à l'ouverture d'une position on fige : beta_e, alpha_e et les niveaux de sortie / retournement.
Pendant le trade le spread surveillé est x_t = log A_t - beta_e log B_t - alpha_e ; le gain
encaissé (avec beta_e) en est exactement la variation, au premier ordre.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def locked_positions(sig, log_a: np.ndarray, log_b: np.ndarray, max_hold: int | np.ndarray,
                     stop_sd: float | None = None, return_trades: bool = False):
    """
    Retourne (positions, betas_verrouillés[, trades]). Les betas verrouillés valent beta_e pendant un trade
    et le beta courant à plat ; ils sont à passer au moteur de gain (qui les décale d'une barre).

    max_hold : durée maximale en barres. Scalaire, ou tableau par barre : on lit alors la valeur à la barre
               d'ENTRÉE et elle est figée pour tout le trade.

    stop_sd : stop adverse en écarts-types stationnaires sigma_inf = sigma/sqrt(2 theta) (calibration de l'entrée)
              AU-DELÀ du niveau d'entrée, mesuré sur le spread verrouillé. None : pas de stop.
              Après un stop on ne ré-entre du même côté que lorsque le signal est repassé à l'intérieur de la bande
              (évite de racheter une paire qui diverge) ; aucun paramètre supplémentaire.
    """
    n = len(log_a)
    pos = np.zeros(n, dtype=np.float32)
    betas = np.array(sig.betas, dtype=float)
    alphas = sig.alphas if sig.alphas is not None else np.zeros(n)
    flip_ok = bool(getattr(sig, "allow_flip", False))
    hold_cap = np.broadcast_to(np.asarray(max_hold, dtype=float), (n,))

    sinf = None
    if sig.theta_hat is not None:                  # nécessaire aussi sans stop : l'excursion adverse est toujours mesurée
        with np.errstate(invalid="ignore", divide="ignore"):
            sinf = np.asarray(sig.sigma_hat) / np.sqrt(2.0 * np.asarray(sig.theta_hat))

    trades = []
    side = 0
    entry_bar = 0
    b_e = a_e = exit_lvl = flip_lvl = 0.0
    entry_lvl = stop_lvl = sinf_e = np.nan
    hold_e = np.inf      # durée maximale du trade en cours, figée à l'entrée
    adverse = 0.0
    blocked = 0          # côté bloqué après un stop (+1 : long, -1 : short)

    def close_trade(i, kind):
        trades.append({"entry": entry_bar, "exit": i, "side": side, "kind": kind, "mae_sd": adverse})

    def open_position(i, s):
        nonlocal side, entry_bar, b_e, a_e, exit_lvl, flip_lvl, entry_lvl, stop_lvl, sinf_e, adverse, hold_e
        side, entry_bar = s, i
        hold_e = hold_cap[i]
        adverse = 0.0
        b_e, a_e = float(sig.betas[i]), float(alphas[i])
        if s == 1:
            exit_lvl, flip_lvl = sig.long_exit_level[i], sig.b_star[i]      # sortie long / entrée short
            entry_lvl = sig.d_star[i]
        else:
            exit_lvl, flip_lvl = sig.short_exit_level[i], sig.d_star[i]     # sortie short / entrée long
            entry_lvl = sig.b_star[i]
        sinf_e = sinf[i] if sinf is not None else np.nan
        stop_lvl = (entry_lvl - s * stop_sd * sinf_e) if stop_sd is not None else np.nan

    for i in range(1, n):
        if blocked == 1 and not sig.entry_long[i]:
            blocked = 0
        if blocked == -1 and not sig.entry_short[i]:
            blocked = 0
        if side == 0:
            if sig.entry_long[i] and blocked != 1:
                open_position(i, 1)
            elif sig.entry_short[i] and blocked != -1:
                open_position(i, -1)
        else:
            x = log_a[i] - b_e * log_b[i] - a_e
            if np.isfinite(sinf_e) and sinf_e > 0:
                adverse = max(adverse, side * (entry_lvl - x) / sinf_e)
            if stop_sd is not None and np.isfinite(stop_lvl) and ((side == 1 and x <= stop_lvl) or (side == -1 and x >= stop_lvl)):
                close_trade(i, "stop")
                blocked, side = side, 0
                pos[i] = 0
                betas[i] = sig.betas[i]
                continue
            if side == 1:
                hit_exit = np.isfinite(exit_lvl) and x >= exit_lvl
                opposite = sig.entry_short[i] > 0
            else:
                hit_exit = np.isfinite(exit_lvl) and x <= exit_lvl
                opposite = sig.entry_long[i] > 0
            timed_out = (i - entry_bar) >= hold_e
            if hit_exit or timed_out:
                if flip_ok and hit_exit and opposite:
                    close_trade(i, "retournement")
                    open_position(i, -side)
                else:
                    close_trade(i, "sortie" if hit_exit else "arrêt temporel")
                    side = 0
        pos[i] = side
        betas[i] = b_e if side != 0 else sig.betas[i]
    if side != 0:
        close_trade(n - 1, "fin de série")
    if return_trades:
        return pos, betas, trades
    return pos, betas
