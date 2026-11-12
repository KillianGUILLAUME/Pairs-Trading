import numpy as np
import pytest

from research.institutional.hjb_solvers import (
    HJBSolverErgodic, HJBSolverFPT, make_solver, mean_exit_time_symmetric_std, mean_fpt_up_std,
)


def _fd_two_sided(a, n=4001):
    y = np.linspace(-a, a, n)
    h = y[1] - y[0]
    A = np.zeros((n, n))
    rhs = -np.ones(n)
    for i in range(1, n - 1):
        A[i, i - 1] = 1 / h ** 2 + y[i] / (2 * h)
        A[i, i] = -2 / h ** 2
        A[i, i + 1] = 1 / h ** 2 - y[i] / (2 * h)
    A[0, 0] = 1; rhs[0] = 0; A[-1, -1] = 1; rhs[-1] = 0
    return np.linalg.solve(A, rhs)[n // 2]


def _fd_one_sided(x, b, L=12.0, n=6001):
    y = np.linspace(-L, b, n)
    h = y[1] - y[0]
    A = np.zeros((n, n))
    rhs = -np.ones(n)
    for i in range(1, n - 1):
        A[i, i - 1] = 1 / h ** 2 + y[i] / (2 * h)
        A[i, i] = -2 / h ** 2
        A[i, i + 1] = 1 / h ** 2 - y[i] / (2 * h)
    A[0, 0] = -1; A[0, 1] = 1; rhs[0] = 0
    A[-1, -1] = 1; rhs[-1] = 0
    return np.interp(x, y, np.linalg.solve(A, rhs))


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
def test_symmetric_exit_time_matches_ode(a):
    assert abs(mean_exit_time_symmetric_std(a) - _fd_two_sided(a)) < 1e-3


@pytest.mark.parametrize("x,b", [(-1.0, 0.0), (-2.0, 0.0), (0.0, 1.5)])
def test_one_sided_passage_time_matches_ode(x, b):
    assert abs(mean_fpt_up_std(x, b) - _fd_one_sided(x, b)) < 1e-3


def test_ergodic_thresholds_monotone_in_cost_sigma_theta():
    base = HJBSolverErgodic(0.1, 0.0, 0.01, 0.001).solve()
    assert base["b_star"] > 0 > base["d_star"]
    assert abs(base["b_star"] + base["d_star"]) < 1e-12          # symétrie autour de μ
    assert HJBSolverErgodic(0.1, 0.0, 0.01, 0.003).solve()["a_star"] > base["a_star"]   # coût ↑ → bande ↑
    assert HJBSolverErgodic(0.1, 0.0, 0.02, 0.001).solve()["a_star"] > base["a_star"]   # σ ↑ → bande ↑
    assert HJBSolverErgodic(0.4, 0.0, 0.01, 0.001).solve()["a_star"] < base["a_star"]   # θ ↑ → bande ↓
    assert not base["at_bound"]
    assert base["expected_cycle_bars"] > 0


def test_ergodic_mu_shift_is_a_translation():
    s0 = HJBSolverErgodic(0.1, 0.0, 0.01, 0.001).solve()
    s1 = HJBSolverErgodic(0.1, 0.05, 0.01, 0.001).solve()
    assert abs((s1["b_star"] - 0.05) - s0["b_star"]) < 1e-12


def test_leung_li_eigenfunctions_solve_the_ode():
    ll = HJBSolverFPT(0.1, 0.0, 0.01, rf=1e-3, cost=0.001)
    xs = np.linspace(-0.03, 0.03, 7)
    h = 1e-5
    for f in (ll.F, ll.G):
        f0 = f(xs); fp = (f(xs + h) - f(xs - h)) / (2 * h); fpp = (f(xs + h) - 2 * f0 + f(xs - h)) / h ** 2
        resid = 0.5 * ll.sigma ** 2 * fpp + ll.theta * (ll.mu - xs) * fp - ll.rf * f0
        assert np.max(np.abs(resid) / np.abs(ll.rf * f0)) < 1e-4


def test_leung_li_levels_widen_as_discount_vanishes():
    prev = None
    for r in (1e-2, 1e-3, 1e-4):
        s = HJBSolverFPT(0.1, 0.0, 0.01, rf=r, cost=0.001).solve()
        assert s["tradable"] and s["d_star"] < 0 < s["b_star"]
        if prev is not None:
            assert s["b_star"] > prev["b_star"] and s["d_star"] < prev["d_star"]
        prev = s


def test_factory_rejects_removed_dp_solver():
    with pytest.raises(ValueError):
        make_solver("dp", 0.1, 0.0, 0.01, 0.001, 1e-3)
    with pytest.raises(ValueError):
        HJBSolverFPT(0.1, 0.0, 0.01, rf=0.0)


# ---------------------------------------------------------------------------
# Solveur ergodique à deux niveaux (entrée a, sortie e)
# ---------------------------------------------------------------------------
from research.institutional.hjb_solvers import _cycle_table, _solve_ergodic_dimensionless, _solve_ergodic_table


def _fd_exit_from(a, x0, n=4001):
    y = np.linspace(-a, a, n)
    h = y[1] - y[0]
    A = np.zeros((n, n))
    rhs = -np.ones(n)
    for i in range(1, n - 1):
        A[i, i - 1] = 1 / h ** 2 + y[i] / (2 * h)
        A[i, i] = -2 / h ** 2
        A[i, i + 1] = 1 / h ** 2 - y[i] / (2 * h)
    A[0, 0] = 1; rhs[0] = 0; A[-1, -1] = 1; rhs[-1] = 0
    return float(np.interp(x0, y, np.linalg.solve(A, rhs)))


@pytest.mark.parametrize("a,e", [(1.0, 0.0), (0.8, 0.5), (1.5, -0.7), (0.6, 0.6)])
def test_cycle_time_table_matches_independent_computation(a, e):
    tb = _cycle_table()
    y = tb["y"]
    i = int(np.argmin(np.abs(tb["a"] - a)))
    j = int(np.argmin(np.abs(y - e)))
    a_g, e_g = float(tb["a"][i]), float(y[j])
    ref = mean_fpt_up_std(-a_g, e_g) + (0.0 if abs(e_g - a_g) < 1e-9 else _fd_exit_from(a_g, e_g))
    assert abs(tb["cycle"][i, j] - ref) / ref < 2e-3


@pytest.mark.parametrize("c", [0.05, 0.2, 0.6])
def test_mean_exit_mode_matches_quadrature_reference(c):
    a_q, r_q, _ = _solve_ergodic_dimensionless(c)
    a_t, e_t, r_t, _, _ = _solve_ergodic_table(c, "mean")
    assert abs(e_t) < 1e-9
    assert abs(r_t - r_q) / r_q < 2e-3
    assert abs(a_t - a_q) < 0.03


@pytest.mark.parametrize("c", [0.02, 0.1, 0.3, 0.6, 1.0, 1.5])
def test_optimal_exit_is_the_flip_and_beats_exit_at_mean(c):
    a, e, r, _, _ = _solve_ergodic_table(c, "optimal")
    assert abs(e - a) < 0.02                      # retournement direct
    r_mean = _solve_ergodic_table(c, "mean")[2]
    assert r >= r_mean - 1e-9


def test_solver_flip_flag_and_levels():
    s = HJBSolverErgodic(0.1, 0.0, 0.01, 0.002).solve()
    assert s["flip"] and abs(s["long_exit"] - s["b_star"]) < 2e-4 * 10
    m = HJBSolverErgodic(0.1, 0.0, 0.01, 0.002, exit_mode="mean").solve()
    assert not m["flip"] and abs(m["long_exit"]) < 1e-12
    assert s["expected_return_per_bar"] >= m["expected_return_per_bar"]
