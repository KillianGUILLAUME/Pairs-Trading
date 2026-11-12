import numpy as np

from research.institutional.hjb_solvers import HJBSolverErgodic, win_probability_std
from research.institutional.signal_engine import InstitutionalSignalEngine
from research.sizing.kelly_fractionnaire import FractionalKelly
from research.sizing.portfolio_manager import PortfolioManager, PortfolioConfig


def test_win_probability_matches_monte_carlo():
    a, e, s = 0.6, 0.6, 1.5
    p_theory = win_probability_std(a, e, s)
    rng = np.random.default_rng(0)
    n, dt = 4000, 0.002
    x = np.full(n, -a)
    alive = np.ones(n, bool)
    win = np.zeros(n, bool)
    for _ in range(40000):
        if not alive.any():
            break
        x[alive] += -x[alive] * dt + np.sqrt(2 * dt) * rng.standard_normal(alive.sum())
        hit_up = alive & (x >= e)
        hit_dn = alive & (x <= -(a + s))
        win |= hit_up
        alive &= ~(hit_up | hit_dn)
    assert abs(win.mean() - p_theory) < 0.03


def test_win_probability_limits():
    assert win_probability_std(0.5, 0.5, 1e-6) < 1e-3          # stop collé à l'entrée : on perd
    assert win_probability_std(0.5, 0.5, 6.0) > 0.99           # stop très loin : on gagne


def test_solver_exports_kelly_inputs():
    s = HJBSolverErgodic(0.1, 0.0, 0.01, 0.002, stop_sd=2.0).solve()
    unit = 0.01 / np.sqrt(0.2)
    assert abs(s["win_return"] - (s["a_star"] + s["e_star"] - 0.002)) < 1e-12
    assert abs(s["loss_return"] - (2.0 * unit + 0.002)) < 1e-12
    assert 0.5 < s["p_win_model"] < 1.0


def test_size_array_accepts_arrays_and_matches_scalar():
    k = FractionalKelly(fraction=0.25, max_leverage=0.5)
    probs = np.array([0.0, 0.6, 0.8, 0.95])
    scalar = k.size_array(probs, 0.02, 0.015)
    vec = k.size_array(probs, np.full(4, 0.02), np.full(4, 0.015))
    np.testing.assert_allclose(scalar, vec)
    assert scalar[0] == 0.0 and np.all(scalar <= 0.5)
    nan = k.size_array(probs, np.array([np.nan, 0.02, 0.02, 0.02]), 0.015)
    assert nan[0] == 0.0 and nan[1] > 0.0


def test_engine_exports_sizing_arrays_and_portfolio_uses_them():
    rng = np.random.default_rng(3)
    n = 3000
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + 3.0
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.9 * sp[t - 1] + 0.004 * rng.standard_normal()
    pa, pb = np.exp(log_b + 0.5 + sp), np.exp(log_b)
    sig = InstitutionalSignalEngine(window=156, hedge_window=156, cost=0.002).generate(np.arange(n), pa, pb)
    assert np.all(np.isnan(sig.win_return[:156]))
    assert np.all(np.isfinite(sig.win_return[160:])) and np.all(sig.win_return[160:] > 0)
    assert np.all((sig.p_win_model[160:] > 0.5) & (sig.p_win_model[160:] < 1.0))
    assert len(sig.slice(200, 400).win_return) == 200
    pm = PortfolioManager(PortfolioConfig(kelly_fraction=0.25, max_gross_exposure=2.0, max_pair_exposure=0.5))
    probs = {"A|B": np.where(np.arange(n) % 100 == 0, 0.8, 0.0)}
    df = pm.allocate_vectorized(probs, {"A|B": np.nan_to_num(sig.win_return)}, {"A|B": np.nan_to_num(sig.loss_return)})
    assert df["A|B"].iloc[0] == 0.0                       # warm-up : pas de gain défini -> taille nulle
    assert (df["A|B"].iloc[200::100] > 0).all()
