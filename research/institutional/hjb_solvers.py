"""
research/institutional/hjb_solvers.py

Seuils optimaux d'entrée / sortie pour un spread Ornstein-Uhlenbeck
    dX_t = θ (μ - X_t) dt + σ dW_t
avec coût de transaction c (fraction du notional par aller-retour, dans l'unité
du spread, i.e. en log-prix).

UNITÉS : θ, σ et r sont exprimés PAR BARRE (la calibration AR(1) de
`signal_engine.calibrate_ou` utilise dt = 1 barre). Un taux annuel doit être
converti avant d'arriver ici (le `InstitutionalSignalEngine` le fait).

Deux solveurs :

1. `HJBSolverErgodic` (par défaut) — critère de Bertram (2010) : maximiser le
   rendement espéré PAR UNITÉ DE TEMPS d'une stratégie symétrique « entrer à μ ∓ a,
   sortir à μ ± e » (a ET e optimisés ; l'optimum est le retournement e = a),
   sans taux d'actualisation. Le cycle moyen est
   la somme de deux temps de premier passage d'un OU (formules fermées par
   quadrature). Après adimensionnement (x̃ = (x-μ)√(2θ)/σ, t̃ = θ t), l'optimum
   ne dépend que de c̃ = c √(2θ)/σ : on le met en cache. Propriétés : a* croît
   avec σ et c, décroît avec θ (a* ≈ C · c^{1/3} σ^{2/3} θ^{-1/3} pour c petit).

2. `HJBSolverFPT` — double arrêt optimal actualisé de Leung & Li (2015),
   « Optimal mean reversion trading with transaction costs and stop-loss exit »
   (IJTAF). Étape 1 (sortie d'un long) : V(x) = sup_τ E_x[e^{-rτ}(X_τ - c)]
   → V(x) = (b*-c) F(x)/F(b*), b* = argmax (b-c)/F(b). Étape 2 (entrée) :
   J(x) = sup_ν E_x[e^{-rν}(V(X_ν) - X_ν - c)] → d* = argmax (V(d)-d-c)/G(d).
   F et G sont les solutions croissante / décroissante de
   (σ²/2) f'' + θ(μ-x) f' - r f = 0, exprimées avec les fonctions de Weber
   (cylindre parabolique) : F(x) = e^{z²/4} D_{-r/θ}(-z), G(x) = e^{z²/4} D_{-r/θ}(z).
   ATTENTION : ce critère exige r > 0 et n'a de sens que si r/θ n'est pas
   négligeable. Avec des barres horaires (θ ~ 0.1 / barre) et r = 5 %/an
   (≈ 5.7e-6 / barre), r/θ ~ 1e-4 : « attendre » est gratuit et les seuils
   partent aux bornes de la grille (aucun trade). C'est pourquoi l'ergodique
   est le défaut pour l'intraday.

Le solveur en différences finies explicite (`hjb_solver_dp.py`, CFL ≈ 17,
instable) a été retiré : voir `legacy/`.
"""

from __future__ import annotations

from functools import lru_cache
from math import erf, exp, pi, sqrt
from typing import Dict

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import pbdv

SQRT_2PI = sqrt(2.0 * pi)


# ============================================================================
# OU standard (θ = 1, σ = √2, μ = 0, variance stationnaire 1) : temps moyens
# de premier passage. Les formules viennent de l'EDO  u'' - y u' = -1.
# ============================================================================

def _Phi(y: float) -> float:
    return 0.5 * (1.0 + erf(y / sqrt(2.0)))


def _integrand_up(y: float) -> float:
    # e^{y²/2} Φ(y) : borné pour y -> -∞ (≈ 1/(√(2π)|y|)), explose pour y grand
    return exp(0.5 * y * y) * _Phi(y)


def mean_fpt_up_std(x: float, b: float) -> float:
    """
    E_x[T_b] pour l'OU standard, x < b (passage vers le HAUT).
    u(x) = √(2π) ∫_x^b e^{y²/2} Φ(y) dy   (solution bornée de u'' - y u' = -1, u(b)=0).
    """
    if x >= b:
        return 0.0
    val, _ = quad(_integrand_up, x, b, limit=200)
    return SQRT_2PI * val


def mean_fpt_down_std(x: float, b: float) -> float:
    """E_x[T_b] pour x > b (passage vers le BAS), par symétrie y -> -y."""
    return mean_fpt_up_std(-x, -b)


def mean_exit_time_symmetric_std(a: float) -> float:
    """
    E_0[T] pour la sortie de (-a, a) en partant de 0 (OU standard).
    u(0) = √(2π) ∫_0^a e^{y²/2} (Φ(y) - 1/2) dy   (u paire, u'(0) = 0, u(±a) = 0).
    """
    if a <= 0:
        return 0.0
    val, _ = quad(lambda y: exp(0.5 * y * y) * (_Phi(y) - 0.5), 0.0, a, limit=200)
    return SQRT_2PI * val


def expected_cycle_time_std(a_tilde: float) -> float:
    """
    Cycle « entrer à -a, sortir à 0, attendre le prochain ±a » (OU standard) :
        T(-a -> 0)  +  E_0[T_{±a}]
    Par symétrie le cycle short est identique.
    """
    return mean_fpt_up_std(-a_tilde, 0.0) + mean_exit_time_symmetric_std(a_tilde)


# ============================================================================
# Solveur ergodique à deux niveaux (Bertram 2010, généralisé aux paires)
# ============================================================================
#
# Stratégie symétrique de paire : entrée LONG à μ - a, sortie à μ + e (e ∈ (-a, a]),
# attente à plat jusqu'à toucher μ ± a, puis entrée du côté touché (SHORT miroir :
# entrée à μ + a, sortie à μ - e). Par symétrie tous les trades sont équivalents :
#     profit moyen par trade = a + e - c
#     durée moyenne          = E[T(-a -> e)] + E_e[sortie de (-a, a)]
# e = a est le RETOURNEMENT direct (toujours investi, sans phase à plat) ;
# e = 0 est l'ancienne règle « sortie à la moyenne ».
# Le taux (a + e - c) / durée ne dépend, en variables adimensionnées, que de
# c̃ = c / σ∞ : la table des durées est calculée une fois, puis chaque résolution
# est un argmax matriciel (~1 ms).
#
# Durées sur l'OU standard (u'' - y u' = -1) via fonction d'échelle S et mesure de
# vitesse m, avec cumul trapézoïdal sur une grille uniforme :
#   S(y) = ∫_0^y e^{t²/2} dt = sqrt(π/2) erfi(y/√2),  m(y) = e^{-y²/2}
#   E_{-a}[T_e]      = √(2π) [K(e) - K(-a)],  K' = e^{y²/2} Φ(y)
#   E_x[sortie(-a,a)] = formule de Green (cumuls P = ∫ S m, Q = ∫ m).

A_TILDE_MAX = 6.0   # borne de recherche en écarts-types stationnaires
_GRID_H = 0.01


@lru_cache(maxsize=1)
def _cycle_table():
    from scipy.special import erfi, ndtr
    h, L = _GRID_H, A_TILDE_MAX
    y = np.arange(-L, L + h / 2, h)
    N = len(y)
    mid = N // 2
    S = sqrt(pi / 2.0) * erfi(y / sqrt(2.0))
    m = np.exp(-y ** 2 / 2.0)
    g = np.exp(y ** 2 / 2.0) * ndtr(y)

    def cum(f):
        c = np.zeros_like(f)
        c[1:] = np.cumsum((f[1:] + f[:-1]) * h / 2.0)
        return c

    P, Q, K = cum(S * m), cum(m), cum(g)
    ia = np.arange(mid + 1, N)[:, None]          # indices de a > 0
    ib = (N - 1 - ia)                            # indices de -a
    ie = np.arange(N)[None, :]                   # indices de e
    valid = (ie > ib) & (ie <= ia)
    T1 = SQRT_2PI * (K[ie] - K[ib])
    den = S[ia] - S[ib]
    tau = ((S[ia] - S[ie]) / den * ((P[ie] - P[ib]) - S[ib] * (Q[ie] - Q[ib]))
           + (S[ie] - S[ib]) / den * (S[ia] * (Q[ia] - Q[ie]) - (P[ia] - P[ie])))
    cycle = np.where(valid, T1 + tau, np.nan)
    a_vals = y[ia[:, 0]]
    profit = a_vals[:, None] + y[None, :]
    return {"y": y, "mid": mid, "ia": ia[:, 0], "a": a_vals, "cycle": cycle, "valid": valid, "profit": profit}


@lru_cache(maxsize=8192)
def _solve_ergodic_table(c_tilde: float, exit_mode: str):
    """argmax_{a, e} (a + e - c̃) / T(a, e). Retourne (ã*, ẽ*, R̃*, T̃*, at_bound)."""
    tb = _cycle_table()
    y, cyc, valid, profit = tb["y"], tb["cycle"], tb["valid"], tb["profit"]
    ok = valid & (profit > c_tilde)
    if exit_mode == "mean":
        ok = ok & (np.abs(y) < _GRID_H / 2)[None, :]
    elif exit_mode == "flip":
        ok = ok & (np.arange(len(y))[None, :] == tb["ia"][:, None])
    elif exit_mode != "optimal":
        raise ValueError(f"exit_mode inconnu : {exit_mode}")
    if not ok.any():
        return float("nan"), float("nan"), float("nan"), float("nan"), True
    with np.errstate(all="ignore"):
        R = np.where(ok, (profit - c_tilde) / cyc, -np.inf)
    k = np.unravel_index(int(np.argmax(R)), R.shape)
    a_t, e_t = float(tb["a"][k[0]]), float(y[k[1]])
    return a_t, e_t, float(R[k]), float(cyc[k]), bool(a_t >= A_TILDE_MAX - 2 * _GRID_H)


@lru_cache(maxsize=4096)
def _solve_ergodic_dimensionless(c_tilde: float) -> tuple[float, float, float]:
    """
    Version « sortie à la moyenne » par quadrature (référence indépendante de la table,
    utilisée par les tests). argmax_{ã > c̃} (ã - c̃) / T̃(ã).
    """
    lo = max(c_tilde, 0.0) + 1e-4
    if lo >= A_TILDE_MAX:
        return float("nan"), float("nan"), float("nan")

    def neg_obj(a: float) -> float:
        T = expected_cycle_time_std(a)
        return -(a - c_tilde) / T if T > 0 else 0.0

    grid = np.linspace(lo, A_TILDE_MAX, 25)
    vals = np.array([neg_obj(a) for a in grid])
    k = int(np.argmin(vals))
    left = grid[max(k - 1, 0)]
    right = grid[min(k + 1, len(grid) - 1)]
    if right <= left:
        right = min(left + 1e-3, A_TILDE_MAX)
    res = minimize_scalar(neg_obj, bounds=(left, right), method="bounded", options={"xatol": 1e-6})
    a_star = float(res.x)
    return a_star, float(-res.fun), expected_cycle_time_std(a_star)


def scale_function_std(y):
    """Fonction d'échelle de l'OU standard : S(y) = ∫_0^y e^{t²/2} dt = sqrt(π/2) erfi(y/√2)."""
    from scipy.special import erfi
    return sqrt(pi / 2.0) * erfi(np.asarray(y, dtype=float) / sqrt(2.0))


def win_probability_std(a_tilde: float, e_tilde: float, stop_tilde: float) -> float:
    """
    P(un long entré à -a atteint +e AVANT le niveau adverse -(a + stop)), OU standard.
    Exacte : la probabilité de sortie par un bord est affine en S (fonction d'échelle).
    """
    lo = -(a_tilde + stop_tilde)
    S_lo, S_in, S_e = scale_function_std([lo, -a_tilde, e_tilde])
    return float((S_in - S_lo) / (S_e - S_lo))


class HJBSolverErgodic:
    """
    Niveaux optimaux (entrée a, sortie e) au sens du rendement par unité de temps.

    exit_mode : "optimal" (défaut, optimise a ET e ; l'optimum est le retournement e = a),
                "flip" (e = a imposé), "mean" (e = 0, ancienne règle).

    solve() -> dict :
        a_star, e_star       : demi-largeur d'entrée et niveau de sortie (unités du spread, relatifs à μ)
        b_star, d_star       : μ + a*, μ - a*  (entrées SHORT / LONG)
        long_exit, short_exit: μ + e*, μ - e*
        flip                 : True si e* = a* (retournement direct, jamais à plat)
        expected_return_per_bar, expected_cycle_bars, at_bound
    """

    def __init__(self, theta: float, mu: float, sigma: float, cost: float = 0.001, exit_mode: str = "optimal",
                 stop_sd: float = 2.0):
        if theta <= 0 or sigma <= 0:
            raise ValueError(f"theta et sigma doivent être > 0 (theta={theta}, sigma={sigma})")
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.cost = float(cost)
        self.exit_mode = exit_mode
        self.stop_sd = float(stop_sd)   # perte adverse supposée, en écarts-types stationnaires au-delà de l'entrée (dimensionnement)
        self.std_lr = self.sigma / sqrt(2.0 * self.theta)   # écart-type stationnaire

    def solve(self) -> Dict[str, float]:
        unit = self.std_lr
        c_tilde = round(self.cost / unit, 4)
        a_t, e_t, r_t, T_t, at_bound = _solve_ergodic_table(c_tilde, self.exit_mode)
        if not np.isfinite(a_t):
            return {
                "a_star": float("nan"), "e_star": float("nan"), "b_star": float("nan"), "d_star": float("nan"),
                "long_exit": float("nan"), "short_exit": float("nan"), "flip": False,
                "expected_return_per_bar": 0.0, "expected_cycle_bars": float("inf"), "at_bound": True,
            }
        a_star, e_star = a_t * unit, e_t * unit
        p_win = win_probability_std(a_t, e_t, self.stop_sd)
        return {
            # Entrées du dimensionnement (Kelly), en fraction du notional par jambe
            "win_return": max(a_star + e_star - self.cost, 0.0),
            "loss_return": self.stop_sd * unit + self.cost,
            "p_win_model": p_win,
            "a_star": a_star,
            "e_star": e_star,
            "b_star": self.mu + a_star,
            "d_star": self.mu - a_star,
            "long_exit": self.mu + e_star,
            "short_exit": self.mu - e_star,
            "flip": bool(e_t >= a_t - 1.5 * _GRID_H),
            "expected_return_per_bar": r_t * unit * self.theta,
            "expected_cycle_bars": T_t / self.theta,
            "at_bound": at_bound,
        }


# ============================================================================
# Solveur actualisé de Leung & Li (2015)
# ============================================================================

class HJBSolverFPT:
    """
    Double arrêt optimal actualisé (Leung & Li 2015). r, θ, σ PAR BARRE.

    solve() -> dict :
        b_star      : niveau de sortie optimal d'un LONG (take-profit)
        d_star      : niveau d'entrée optimal d'un LONG
        short_entry : 2μ - d*   (par symétrie de l'OU autour de μ)
        short_exit  : 2μ - b*
        long_exit   : b*  (alias)
        value_exit, value_entry : V(d*) et J(d*)
        at_bound    : True si b* ou d* a touché une borne (r/θ trop petit)
        tradable    : False si aucune entrée n'est rentable (J ≤ 0)
    """

    Z_CLIP = 30.0
    SEARCH_SD = 8.0

    def __init__(self, theta: float, mu: float, sigma: float, rf: float = 1e-3, cost: float = 0.001):
        if theta <= 0 or sigma <= 0:
            raise ValueError(f"theta et sigma doivent être > 0 (theta={theta}, sigma={sigma})")
        if rf <= 0:
            raise ValueError(
                "HJBSolverFPT exige r > 0 (par barre). Avec r = 0 le problème actualisé "
                "est dégénéré (attendre est gratuit) ; utiliser HJBSolverErgodic."
            )
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.rf = float(rf)
        self.cost = float(cost)
        self.std_lr = self.sigma / sqrt(2.0 * self.theta)
        self._nu = -self.rf / self.theta                     # ordre de D_ν

    # -- fonctions propres -------------------------------------------------
    def _z(self, x):
        z = (np.asarray(x, dtype=float) - self.mu) * sqrt(2.0 * self.theta) / self.sigma
        return np.clip(z, -self.Z_CLIP, self.Z_CLIP)

    def F(self, x):
        """Solution croissante : e^{z²/4} D_{-r/θ}(-z)."""
        z = self._z(x)
        val, _ = pbdv(self._nu, -z)
        return np.maximum(val, 1e-300) * np.exp(z * z / 4.0)

    def G(self, x):
        """Solution décroissante : e^{z²/4} D_{-r/θ}(z)."""
        z = self._z(x)
        val, _ = pbdv(self._nu, z)
        return np.maximum(val, 1e-300) * np.exp(z * z / 4.0)

    def _phi(self, x):          # compat avec l'ancien code
        return self.F(x)

    # -- étape 1 : sortie --------------------------------------------------
    def _exit_ratio(self, b: float) -> float:
        payoff = b - self.cost
        if payoff <= 0:
            return -1e300
        return payoff / float(self.F(b))

    def solve_exit(self) -> tuple[float, bool]:
        lo = max(self.mu, self.cost) + 1e-9
        hi = self.mu + self.SEARCH_SD * self.std_lr
        if hi <= lo:
            hi = lo + self.std_lr
        res = minimize_scalar(lambda b: -self._exit_ratio(b), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-9})
        b_star = float(res.x)
        return b_star, bool(b_star >= hi - 1e-6 * self.std_lr)

    # -- étape 2 : entrée --------------------------------------------------
    def value_exit(self, x, b_star: float):
        """V(x) = (b*-c) F(x)/F(b*) pour x < b*, (x - c) sinon."""
        x = np.asarray(x, dtype=float)
        cont = (b_star - self.cost) * self.F(x) / float(self.F(b_star))
        return np.where(x < b_star, cont, x - self.cost)

    def _entry_ratio(self, d: float, b_star: float) -> float:
        payoff = float(self.value_exit(d, b_star)) - d - self.cost
        if payoff <= 0:
            return -1e300
        return payoff / float(self.G(d))

    def solve_entry(self, b_star: float) -> tuple[float, bool, float]:
        lo = self.mu - self.SEARCH_SD * self.std_lr
        hi = b_star - 1e-9
        res = minimize_scalar(lambda d: -self._entry_ratio(d, b_star), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-9})
        d_star = float(res.x)
        best = -float(res.fun)
        tradable = best > 0
        return d_star, bool(d_star <= lo + 1e-6 * self.std_lr), (best * float(self.G(d_star)) if tradable else 0.0)

    def solve(self) -> Dict[str, float]:
        b_star, b_bound = self.solve_exit()
        d_star, d_bound, j_val = self.solve_entry(b_star)
        tradable = j_val > 0
        return {
            "b_star": b_star,
            "d_star": d_star if tradable else float("nan"),
            "long_exit": b_star,
            "short_entry": 2.0 * self.mu - d_star if tradable else float("nan"),
            "short_exit": 2.0 * self.mu - b_star,
            "value_exit": float(self.value_exit(d_star, b_star)),
            "value_entry": j_val,
            "at_bound": bool(b_bound or d_bound),
            "tradable": bool(tradable),
        }


# ============================================================================
# Fabrique
# ============================================================================

def make_solver(solver_type: str, theta: float, mu: float, sigma: float, cost: float, r_per_bar: float,
                exit_mode: str = "optimal", stop_sd: float = 2.0):
    st = solver_type.lower()
    if st == "ergodic":
        return HJBSolverErgodic(theta, mu, sigma, cost=cost, exit_mode=exit_mode, stop_sd=stop_sd)
    if st in ("fpt", "leung_li", "discounted"):
        return HJBSolverFPT(theta, mu, sigma, rf=r_per_bar, cost=cost)
    if st == "dp":
        raise ValueError("Le solveur 'dp' (différences finies explicites, instable) a été retiré ; "
                         "utiliser 'ergodic' ou 'fpt'. Ancien code dans legacy/.")
    raise ValueError(f"solver_type inconnu : {solver_type}")


def thresholds_from_solution(sol: Dict[str, float], mu: float) -> Dict[str, float]:
    """
    Normalise la sortie des deux solveurs en quatre niveaux :
        long_entry, long_exit, short_entry, short_exit  (NaN si non tradable).
    """
    if "a_star" in sol:                                  # ergodique
        return {
            "long_entry": sol["d_star"], "long_exit": sol["long_exit"],
            "short_entry": sol["b_star"], "short_exit": sol["short_exit"],
        }
    return {                                             # Leung-Li
        "long_entry": sol["d_star"], "long_exit": sol["long_exit"],
        "short_entry": sol["short_entry"], "short_exit": sol["short_exit"],
    }


if __name__ == "__main__":
    theta, mu, sigma, cost = 0.1, 0.0, 0.01, 0.001
    e = HJBSolverErgodic(theta, mu, sigma, cost).solve()
    print(f"Ergodique : a*={e['a_star']:.5f} ({e['a_star']/(sigma/sqrt(2*theta)):.2f} σ∞), "
          f"cycle={e['expected_cycle_bars']:.1f} barres, rendement/barre={e['expected_return_per_bar']:.2e}")
    for r in (1e-2, 1e-3, 0.05 / 8760):
        ll = HJBSolverFPT(theta, mu, sigma, rf=r, cost=cost).solve()
        print(f"Leung-Li r={r:.2e} : d*={ll['d_star']:.5f} b*={ll['b_star']:.5f} tradable={ll['tradable']} at_bound={ll['at_bound']}")
