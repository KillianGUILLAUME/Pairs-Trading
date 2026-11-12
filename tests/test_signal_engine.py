import numpy as np
import warnings

from research.backtest.engine import BacktestConfig, BacktestEngine
from research.institutional.kalman_filter import KalmanHedgeRatio
from research.institutional.signal_engine import InstitutionalSignalEngine


def _cointegrated_pair(seed=3, n=4000):
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + 3.0
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.9 * sp[t - 1] + 0.004 * rng.standard_normal()
    log_a = log_b + 0.5 + sp
    return np.exp(log_a), np.exp(log_b), sp


def test_kalman_returns_a_priori_innovation():
    pa, pb, sp = _cointegrated_pair()
    kh = KalmanHedgeRatio()
    innov, post = [], []
    for i in range(len(pa)):
        b, a, e, S = kh.update(np.log(pb[i]), np.log(pa[i]))
        innov.append(e)
        post.append(np.log(pa[i]) - b * np.log(pb[i]) - a)
    innov, post = np.array(innov[500:]), np.array(post[500:])
    assert np.std(innov) > np.std(post)           # le résidu a posteriori est rétréci
    assert np.corrcoef(innov, sp[500:])[0, 1] > 0.3        # bruit d'estimation du beta Kalman inclus


def test_engine_runs_without_failures_and_trades():
    pa, pb, _ = _cointegrated_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = InstitutionalSignalEngine(window=156, hedge_window=156, solver_type="ergodic", cost=0.002).generate(np.arange(len(pa)), pa, pb)
    d = sig.diagnostics
    assert d["n_solver_failures"] == 0 and d["n_recalibrations"] > 0
    assert d["n_entry_long"] > 0 and d["n_entry_short"] > 0
    assert np.all(np.isnan(sig.b_star[:156]))
    assert np.nanmin(sig.b_star - sig.mu_ref) > 0 and np.nanmax(sig.d_star - sig.mu_ref) < 0
    res = BacktestEngine(BacktestConfig()).run(sig)
    assert res["metrics"]["n_trades"] > 0
    sliced = sig.slice(1000, 2000)
    assert len(sliced.exit_long) == 1000


def test_rf_is_annual_and_converted_per_bar():
    eng = InstitutionalSignalEngine(rf=0.05, bars_per_year=8760)
    assert abs(eng.r_per_bar - 0.05 / 8760) < 1e-15


def _beta_pair(beta=1.5, seed=5, n=5000, theta=0.02):
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + 3.0
    sp = np.zeros(n)
    b = np.exp(-theta)
    for t in range(1, n):
        sp[t] = b * sp[t - 1] + 0.006 * rng.standard_normal()
    return np.exp(beta * log_b + 0.5 + sp), np.exp(log_b), np.exp(-theta)


def test_rolling_ols_hedge_preserves_spread_dynamics_while_kalman_distorts_it():
    pa, pb, phi_true = _beta_pair()
    ts = np.arange(len(pa))

    def phi_of(sig, start=1200):
        s = sig.spreads[start:]
        return np.polyfit(s[:-1], s[1:], 1)[0]

    ols = InstitutionalSignalEngine(hedge="rolling_ols").generate(ts, pa, pb)
    kal = InstitutionalSignalEngine(hedge="kalman").generate(ts, pa, pb)
    assert abs(phi_of(ols) - phi_true) < 0.03          # dynamique préservée
    assert phi_of(kal) < 0.85                          # innovation de Kalman : filtre passe-haut
    assert abs(np.nanmedian(ols.betas[1200:]) - 1.5) < 0.05


def test_rolling_ols_uses_only_past_data():
    pa, pb, _ = _beta_pair(n=1500)
    ts = np.arange(1500)
    full = InstitutionalSignalEngine(window=156, hedge="rolling_ols", hedge_window=300).generate(ts, pa, pb)
    cut = InstitutionalSignalEngine(window=156, hedge="rolling_ols", hedge_window=300).generate(ts[:1000], pa[:1000], pb[:1000])
    np.testing.assert_allclose(full.betas[:1000], cut.betas, rtol=0, atol=1e-12)
    np.testing.assert_allclose(full.spreads[:1000], cut.spreads, rtol=0, atol=1e-12)


def test_thresholds_anchor_on_calibrated_mean_not_on_the_ema():
    pa, pb, _ = _beta_pair(n=2500)
    sig = InstitutionalSignalEngine(window=312, hedge="rolling_ols", hedge_window=300).generate(np.arange(2500), pa, pb)
    mu = sig.mu_ref[400:]
    # μ calibré : constant par morceaux (recalibration toutes les 8 barres), pas une EMA qui suit le spread
    changes = np.count_nonzero(np.diff(mu))
    assert changes <= len(mu) / 8 + 2


def test_time_stop_can_be_disabled():
    from research.backtest.engine import BacktestConfig, BacktestEngine
    pa, pb, _ = _beta_pair(n=2500)
    sig = InstitutionalSignalEngine(window=312, hedge="rolling_ols", hedge_window=300).generate(np.arange(2500), pa, pb)
    r = BacktestEngine(BacktestConfig(time_stop_mult=None)).run(sig)
    assert r["metrics"]["n_trades"] > 0
