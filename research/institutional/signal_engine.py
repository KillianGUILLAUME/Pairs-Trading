"""
research/institutional/signal_engine.py

Moteur de signaux « institutionnel » : Kalman (hedge ratio dynamique) → spread =
innovation → calibration AR(1) → OU → seuils optimaux (solveur HJB) → signaux.

Unités : 1 barre = 1 unité de temps (θ, σ par barre). Le taux `rf` est ANNUEL et
converti en taux par barre via `bars_per_year` (8760 pour du 1h).

Solveurs (voir `hjb_solvers.py`) :
    - "ergodic" (défaut) : bandes symétriques μ ± a*, sortie à μ, critère de
      rendement par unité de temps (Bertram 2010). Pas de taux d'actualisation.
    - "fpt" : double arrêt optimal actualisé de Leung & Li (2015) ; entrée long à
      d*, sortie long à b*, short par symétrie. Dégénère quand r/θ → 0.

`cost` est le coût TOTAL d'un aller-retour sur la paire, en unités du spread
(log-prix) : 4 jambes × (frais + slippage) ≈ 4 × (4 + 1) bps = 0.002 avec les
défauts de `BacktestConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

from research.institutional.hjb_solvers import make_solver, thresholds_from_solution
from research.institutional.kalman_filter import KalmanHedgeRatio, KalmanOUState


@dataclass
class InstitutionalSignal:
    timestamps: np.ndarray
    price_a: np.ndarray
    price_b: np.ndarray
    spreads: np.ndarray            # innovation de Kalman (spread centré)
    zscores: np.ndarray            # (spread - μ) / std glissant CAUSAL
    betas: np.ndarray              # hedge ratio a posteriori
    entry_long: np.ndarray
    entry_short: np.ndarray
    exit_signal: np.ndarray        # compat : croisement de μ (indépendant du côté)
    b_star: np.ndarray             # niveau d'entrée SHORT (μ + a*)
    d_star: np.ndarray             # niveau d'entrée LONG  (μ - a*)
    exit_long: Optional[np.ndarray] = None    # sortie d'un LONG (spread >= niveau de sortie long)
    exit_short: Optional[np.ndarray] = None   # sortie d'un SHORT (spread <= niveau de sortie short)
    win_return: Optional[np.ndarray] = None   # gain attendu par trade, fraction du notional par jambe (a + e - c), NaN avant warm-up
    loss_return: Optional[np.ndarray] = None  # perte adverse supposée par trade (stop_sd * sigma_inf + c)
    p_win_model: Optional[np.ndarray] = None  # P(cible atteinte avant la perte adverse), fonction d'échelle de l'OU
    cycle_bars: Optional[np.ndarray] = None   # durée moyenne d'un cycle, en barres
    allow_flip: bool = False                  # True : retournement direct long <-> short (solveur ergodique optimal)
    mus: Optional[np.ndarray] = None
    alphas: Optional[np.ndarray] = None       # intercept de couverture par barre (spread = log A - beta log B - alpha)
    theta_hat: Optional[np.ndarray] = None    # θ de la dernière calibration (par barre), diagnostic
    sigma_hat: Optional[np.ndarray] = None    # σ de la dernière calibration (par barre), diagnostic
    mu_ref: Optional[np.ndarray] = None       # référence des seuils (EMA ou μ calibré selon mean_source)
    long_exit_level: Optional[np.ndarray] = None
    short_exit_level: Optional[np.ndarray] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def slice(self, start: int, end: int) -> "InstitutionalSignal":
        def _s(a):
            return None if a is None else a[start:end]
        return InstitutionalSignal(
            timestamps=self.timestamps[start:end],
            price_a=self.price_a[start:end],
            price_b=self.price_b[start:end],
            spreads=self.spreads[start:end],
            zscores=self.zscores[start:end],
            betas=self.betas[start:end],
            entry_long=self.entry_long[start:end],
            entry_short=self.entry_short[start:end],
            exit_signal=self.exit_signal[start:end],
            b_star=self.b_star[start:end],
            d_star=self.d_star[start:end],
            exit_long=_s(self.exit_long),
            exit_short=_s(self.exit_short),
            win_return=_s(self.win_return),
            loss_return=_s(self.loss_return),
            p_win_model=_s(self.p_win_model),
            cycle_bars=_s(self.cycle_bars),
            allow_flip=self.allow_flip,
            mus=_s(self.mus),
            mu_ref=_s(self.mu_ref),
            alphas=_s(self.alphas),
            theta_hat=_s(self.theta_hat),
            sigma_hat=_s(self.sigma_hat),
            long_exit_level=_s(self.long_exit_level),
            short_exit_level=_s(self.short_exit_level),
            diagnostics=dict(self.diagnostics),
        )


class InstitutionalSignalEngine:
    """
    Moteur de génération de signaux V2 (Kalman + OU + HJB).
    """

    def __init__(
        self,
        window: int = 624,
        solver_type: str = "ergodic",
        rf: float = 0.05,
        bars_per_year: float = 8760.0,
        cost: float = 0.002,
        use_kalman: bool = False,
        recalib_freq: int = 8,
        kalman_delta: float = 1e-5,
        kalman_R: float = 1e-3,
        mu_q: float = 1e-4,
        mu_r: float = 1e-2,
        min_theta: float = 1e-4,
        exit_mode: str = "optimal",
        stop_sd: float = 2.0,
        oracle_params: Optional[Callable[[int], Tuple[float, float, float]]] = None,
        fixed_beta: Optional[float] = None,
        mean_source: str = "calibrated",
        hedge: str = "rolling_ols",
        hedge_window: int = 500,
        hedge_refresh: int = 24,
    ):
        self.window = int(window)
        self.solver_type = solver_type
        self.rf = float(rf)                       # taux ANNUEL
        self.bars_per_year = float(bars_per_year)
        self.r_per_bar = self.rf / self.bars_per_year
        self.cost = float(cost)
        self.use_kalman = use_kalman
        self.recalib_freq = int(recalib_freq)
        self.kalman_delta = kalman_delta
        self.kalman_R = kalman_R
        self.mu_q = mu_q
        self.mu_r = mu_r
        self.min_theta = min_theta
        # Référence des seuils : "ema" = moyenne exponentielle rapide du spread (historique, filtre passe-haut),
        # "calibrated" = μ de la dernière calibration AR(1), constant entre deux recalibrations (cohérent avec l'OU).
        if mean_source not in ("ema", "calibrated"):
            raise ValueError("mean_source doit être 'ema' ou 'calibrated'")
        self.mean_source = mean_source
        # Couche de couverture : "kalman" (innovation, historique) ; "rolling_ols" : (β, α) par OLS avec constante
        # sur les `hedge_window` barres STRICTEMENT passées, rafraîchis toutes les `hedge_refresh` barres et
        # constants entre deux rafraîchissements ; `fixed_beta` (si fourni) prime sur les deux.
        if hedge not in ("kalman", "rolling_ols"):
            raise ValueError("hedge doit être 'kalman' ou 'rolling_ols'")
        self.hedge = hedge
        self.hedge_window = int(hedge_window)
        self.hedge_refresh = int(hedge_refresh)
        self.fixed_beta = fixed_beta   # None : hedge ratio Kalman ; sinon spread = log A - fixed_beta log B (sans Kalman)
        self.oracle_params = oracle_params   # test uniquement : i -> (theta, mu, sigma) VRAIS, remplace la calibration
        self.stop_sd = stop_sd          # perte adverse supposée (écarts-types stationnaires) : dimensionnement seulement
        self.exit_mode = exit_mode      # solveur ergodique : "optimal" | "flip" | "mean"

    # ------------------------------------------------------------------
    def calibrate_ou(self, spread_window: np.ndarray):
        """
        Calibre (θ, μ, σ) PAR BARRE via régression AR(1) : x_{t+1} = a + b x_t + ε.
            θ = -ln(b),  μ = a / (1 - b),  σ = std(ε) · sqrt(-2 ln(b) / (1 - b²)).
        """
        spread_window = np.asarray(spread_window, dtype=float)
        if len(spread_window) < 10:
            return 0.1, float(np.mean(spread_window)), float(np.std(spread_window))

        y = spread_window[1:]
        x = sm.add_constant(spread_window[:-1])
        res = sm.OLS(y, x).fit()

        a, b = res.params
        residuals = res.resid

        dt = 1.0  # 1 barre = 1 unité de temps
        b = float(np.clip(b, 0.0001, 0.9999))

        theta = -np.log(b) / dt
        mu = a / (1 - b)
        sigma = np.std(residuals) * np.sqrt(-2 * np.log(b) / (dt * (1 - b ** 2)))

        return float(theta), float(mu), float(sigma)

    # ------------------------------------------------------------------
    def generate(self, timestamps, price_a, price_b, betas=None) -> InstitutionalSignal:
        n = len(price_a)
        log_a = np.log(np.asarray(price_a, dtype=float))
        log_b = np.log(np.asarray(price_b, dtype=float))

        kh = KalmanHedgeRatio(delta=self.kalman_delta, R=self.kalman_R)
        ko = KalmanOUState(q_mu=self.mu_q, r_obs=self.mu_r)

        dynamic_betas = np.zeros(n)
        alphas = np.zeros(n)
        spreads = np.zeros(n)
        mus = np.zeros(n)

        entry_long = np.zeros(n)
        entry_short = np.zeros(n)
        exit_signal = np.zeros(n)
        exit_long = np.zeros(n)
        exit_short = np.zeros(n)

        win_arr = np.full(n, np.nan)
        loss_arr = np.full(n, np.nan)
        pwin_arr = np.full(n, np.nan)
        cyc_arr = np.full(n, np.nan)
        sizing = None
        mu_ref_arr = np.full(n, np.nan)
        theta_arr = np.full(n, np.nan)
        sigma_arr = np.full(n, np.nan)
        theta_cur = sigma_cur = np.nan
        ols_beta = None
        ols_alpha = 0.0
        first_valid = max(self.window, self.hedge_window) if (self.hedge == "rolling_ols" and self.fixed_beta is None) else self.window
        long_entry_lvl = np.full(n, np.nan)
        short_entry_lvl = np.full(n, np.nan)
        long_exit_lvl = np.full(n, np.nan)
        short_exit_lvl = np.full(n, np.nan)

        # Décalages des niveaux par rapport à μ (appliqués au μ dynamique entre deux recalibrations)
        offsets = None
        mu_cal = np.nan
        n_recal = n_fail = n_bound = n_nan_bars = 0
        flip_votes = 0
        warned_bound = False
        last_error = None

        for i in range(n):
            # 1. Kalman : hedge ratio + INNOVATION (résidu a priori) = spread
            alpha = 0.0
            if self.fixed_beta is not None:
                beta, innovation = self.fixed_beta, log_a[i] - self.fixed_beta * log_b[i]
            elif self.hedge == "rolling_ols":
                if i >= self.hedge_window and (i - self.hedge_window) % self.hedge_refresh == 0:
                    lb, la = log_b[i - self.hedge_window:i], log_a[i - self.hedge_window:i]
                    vb = lb - lb.mean()
                    denom = float(vb @ vb)
                    if denom > 1e-12:
                        first_fit = ols_beta is None
                        ols_beta = float(vb @ (la - la.mean())) / denom
                        ols_alpha = float(la.mean() - ols_beta * lb.mean())
                        if first_fit:
                            # Première couverture : on recalcule l'historique avec ces paramètres (estimés sur les
                            # barres < i seulement), pour que les fenêtres de calibration ne mélangent pas deux
                            # définitions du spread. Aucune décision n'est prise avant `first_valid`.
                            spreads[:i] = log_a[:i] - ols_beta * log_b[:i] - ols_alpha
                            dynamic_betas[:i] = ols_beta
                            alphas[:i] = ols_alpha
                            ko = KalmanOUState(q_mu=self.mu_q, r_obs=self.mu_r)
                            for j in range(i):
                                mus[j] = ko.update(spreads[j])
                if ols_beta is None:          # avant la première couverture : spread brut, aucun signal
                    beta, innovation = 1.0, log_a[i] - log_b[i]
                else:
                    beta, alpha, innovation = ols_beta, ols_alpha, log_a[i] - ols_beta * log_b[i] - ols_alpha
            else:
                beta, alpha, innovation, _ = kh.update(log_b[i], log_a[i])
            dynamic_betas[i] = beta
            alphas[i] = alpha
            spreads[i] = innovation

            # 2. Équilibre dynamique μ (lissage adaptatif du spread)
            mus[i] = ko.update(spreads[i])

            if i < first_valid:
                continue

            # 3. Recalibration périodique OU + solveur HJB (fenêtre causale, barre courante incluse)
            if (i - self.window) % self.recalib_freq == 0:
                window_slice = spreads[i - self.window + 1 : i + 1]
                try:
                    if self.oracle_params is not None:
                        theta, mu_ar, sigma = self.oracle_params(i)
                        target_mu = mu_ar
                    else:
                        theta, mu_ar, sigma = self.calibrate_ou(window_slice)
                        target_mu = mus[i] if self.use_kalman else mu_ar
                    if np.isfinite(theta) and theta > self.min_theta and np.isfinite(sigma) and sigma > 0:
                        mu_cal = mu_ar
                        theta_cur, sigma_cur = theta, sigma
                        sol = make_solver(self.solver_type, theta, target_mu, sigma, self.cost, self.r_per_bar,
                                          exit_mode=self.exit_mode, stop_sd=self.stop_sd).solve()
                        sizing = (sol.get("win_return", np.nan), sol.get("loss_return", np.nan),
                                  sol.get("p_win_model", np.nan), sol.get("expected_cycle_bars", np.nan))
                        flip_votes += int(bool(sol.get("flip", False)))
                        lv = thresholds_from_solution(sol, target_mu)
                        offsets = {k: (v - target_mu) for k, v in lv.items()}
                        n_recal += 1
                        if sol.get("at_bound", False):
                            n_bound += 1
                            if not warned_bound:
                                logger.warning(
                                    f"[{self.solver_type}] seuil en butée de grille à la barre {i} "
                                    f"(theta={theta:.4f}, sigma={sigma:.5f}, r/theta={self.r_per_bar/theta:.1e}) : "
                                    "r/θ trop faible pour le critère actualisé ? (voir hjb_solvers.py)"
                                )
                                warned_bound = True
                    else:
                        n_fail += 1
                        last_error = f"calibration non exploitable (theta={theta}, sigma={sigma})"
                except Exception as e:  # noqa: BLE001 - on compte et on continue, sans masquer
                    n_fail += 1
                    last_error = repr(e)
                    logger.debug(f"Solveur HJB en échec à la barre {i} : {e!r}")

            if offsets is None:
                n_nan_bars += 1
                continue

            if sizing is not None:
                win_arr[i], loss_arr[i], pwin_arr[i], cyc_arr[i] = sizing
            mu_i = mus[i] if self.mean_source == "ema" else mu_cal
            mu_ref_arr[i] = mu_i
            theta_arr[i], sigma_arr[i] = theta_cur, sigma_cur
            long_entry_lvl[i] = mu_i + offsets["long_entry"]
            short_entry_lvl[i] = mu_i + offsets["short_entry"]
            long_exit_lvl[i] = mu_i + offsets["long_exit"]
            short_exit_lvl[i] = mu_i + offsets["short_exit"]
            if not (np.isfinite(long_entry_lvl[i]) and np.isfinite(short_entry_lvl[i])):
                n_nan_bars += 1

            # 4. Signaux (les comparaisons avec NaN valent False -> pas de signal)
            s_i = spreads[i]
            entry_long[i] = float(s_i < long_entry_lvl[i])
            entry_short[i] = float(s_i > short_entry_lvl[i])
            exit_long[i] = float(s_i >= long_exit_lvl[i])
            exit_short[i] = float(s_i <= short_exit_lvl[i])

            # 5. Compat : sortie au croisement de μ (indépendante du côté)
            if (spreads[i - 1] > mus[i - 1] and s_i <= mu_i) or (spreads[i - 1] < mus[i - 1] and s_i >= mu_i):
                exit_signal[i] = 1.0

        # Z-score CAUSAL : écart-type glissant du spread sur `window` barres
        roll_std = pd.Series(spreads).rolling(self.window, min_periods=max(10, self.window // 2)).std().to_numpy()
        zscores = (spreads - mus) / (roll_std + 1e-9)

        diagnostics = {
            "n_bars": n,
            "n_recalibrations": n_recal,
            "n_solver_failures": n_fail,
            "n_at_bound": n_bound,
            "n_bars_without_thresholds": n_nan_bars,
            "n_entry_long": int(entry_long.sum()),
            "n_entry_short": int(entry_short.sum()),
            "solver_type": self.solver_type,
            "r_per_bar": self.r_per_bar,
            "cost": self.cost,
        }
        if n_fail:
            logger.warning(f"Signal engine : {n_fail} recalibration(s) en échec sur {n_recal + n_fail} "
                           f"(dernière erreur : {last_error}).")
        if n_nan_bars and n_recal == 0:
            logger.warning("Signal engine : aucun seuil valide sur toute la série, aucun signal émis.")

        return InstitutionalSignal(
            timestamps=timestamps,
            price_a=price_a,
            price_b=price_b,
            spreads=spreads,
            zscores=zscores,
            betas=dynamic_betas,
            entry_long=entry_long,
            entry_short=entry_short,
            exit_signal=exit_signal,
            b_star=short_entry_lvl,
            d_star=long_entry_lvl,
            exit_long=exit_long,
            exit_short=exit_short,
            win_return=win_arr,
            loss_return=loss_arr,
            p_win_model=pwin_arr,
            cycle_bars=cyc_arr,
            allow_flip=bool(n_recal and flip_votes > n_recal / 2),
            mus=mus,
            mu_ref=mu_ref_arr,
            alphas=alphas,
            theta_hat=theta_arr,
            sigma_hat=sigma_arr,
            long_exit_level=long_exit_lvl,
            short_exit_level=short_exit_lvl,
            diagnostics=diagnostics,
        )
