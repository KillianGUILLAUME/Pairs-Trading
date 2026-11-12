import numpy as np

from research.backtest.engine import BacktestConfig, CostModel, PnLEngine, PerformanceMetrics, PositionManager
from research.backtest.stress_test import FeeScenario, RealisticCostModel, RealisticPnLEngine


def _three_bar_long():
    pa = np.array([100, 100, 101, 102.01, 103.03, 103.03, 103.03])
    pb = np.full(7, 100.0)
    pos = np.array([0, 1, 1, 1, 0, 0, 0], dtype=float)
    return pa, pb, pos, np.ones(7)


def test_exit_bar_pnl_is_credited():
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, position_size=1.0)
    pa, pb, pos, betas = _three_bar_long()
    df = PnLEngine(cfg, CostModel(0.0, 0.0)).compute(pos, np.zeros(7), pa, pb, betas)
    # position portée pendant les barres 2, 3 et 4 (décision en 1, 2, 3) -> trois barres créditées
    assert df.raw_pnl.iloc[2] > 0 and df.raw_pnl.iloc[3] > 0 and df.raw_pnl.iloc[4] > 0
    # notional figé à l'entrée (100k) : 3 barres à ~1 % -> ~3 000 de PnL
    assert df.raw_pnl.sum() > 2_990
    assert abs(df.capital.iloc[-1] - (100_000 + df.raw_pnl.sum())) < 1e-6


def test_realistic_engine_credits_exit_bar_too():
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, position_size=1.0)
    pa, pb, pos, betas = _three_bar_long()
    sc = FeeScenario(name="zero", fee_rate=0.0, slippage_bps=0.0, funding_rate_8h=0.0)
    df = RealisticPnLEngine(cfg, RealisticCostModel(sc)).compute(pos, np.zeros(7), pa, pb, betas)
    assert df.raw_pnl.iloc[4] > 0


def test_n_trades_counts_entries_only():
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0)
    pa, pb, pos, betas = _three_bar_long()
    df = PnLEngine(cfg, CostModel(0.0, 0.0)).compute(pos, np.zeros(7), pa, pb, betas)
    assert PerformanceMetrics.compute(df)["n_trades"] == 1


def test_side_aware_exits():
    pm = PositionManager(BacktestConfig(cooldown_bars=0))
    entry_long = np.array([0, 1, 0, 0, 0, 0, 0], dtype=float)
    entry_short = np.zeros(7)
    exit_signal = np.zeros(7)
    exit_long = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    exit_short = np.array([0, 0, 1, 0, 0, 0, 0], dtype=float)   # ne doit PAS fermer un long
    pos = pm.compute_positions(entry_long, entry_short, exit_signal, 100, exit_long=exit_long, exit_short=exit_short)
    assert pos.tolist() == [0, 1, 1, 0, 0, 0, 0]


def test_flip_goes_directly_from_long_to_short():
    pm = PositionManager(BacktestConfig(cooldown_bars=0))
    entry_long = np.array([0, 1, 0, 0, 0, 1, 0], dtype=float)
    entry_short = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    exit_signal = np.zeros(7)
    pos = pm.compute_positions(entry_long, entry_short, exit_signal, 100, allow_flip=True)
    assert pos.tolist() == [0, 1, 1, -1, -1, 1, 1]
    pos_no = pm.compute_positions(entry_long, entry_short, exit_signal, 100, allow_flip=False)
    assert pos_no.tolist() == [0, 1, 1, 1, 1, 1, 1]


def test_locked_hedge_makes_signal_move_equal_realised_gain():
    from research.backtest.engine import BacktestEngine
    from research.institutional.signal_engine import InstitutionalSignalEngine
    rng = np.random.default_rng(7)
    n = 4000
    # beta réel qui dérive : le rafraîchissement OLS crée des sauts de spread-signal non encaissables
    log_b = np.cumsum(0.01 * rng.standard_normal(n)) + 3.0
    beta_t = 1.0 + 0.5 * np.sin(np.arange(n) / 700.0)
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.97 * sp[t - 1] + 0.006 * rng.standard_normal()
    pa, pb = np.exp(beta_t * log_b + 0.5 + sp), np.exp(log_b)
    sig = InstitutionalSignalEngine(window=400, hedge_window=400, hedge_refresh=24, cost=0.002).generate(np.arange(n), pa, pb)
    cfg = BacktestConfig(position_size=0.1, fee_rate=0.0, slippage_bps=0.0)
    res_lock = BacktestEngine(cfg).run(sig, lock_hedge=True)
    pos = res_lock["positions"]
    assert (pos != 0).any()
    df = res_lock["df"]
    # pendant un trade, le beta utilisé par le moteur de gain est constant (verrouillé à l'entrée)
    beta_r = df["beta_r"].to_numpy()
    for i in range(1, n - 1):
        if pos[i] != 0 and pos[i + 1] == pos[i] and pos[i] == pos[i - 1]:
            assert beta_r[i + 1] == beta_r[i] or abs(beta_r[i + 1] - beta_r[i]) < 1e-12
