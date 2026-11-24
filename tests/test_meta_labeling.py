"""
Chaîne HJB -> XGB : trade, label net, features, filtre par trade, découpage purgé, chargement de l'oracle.

Ne pas importer xgboost ni sklearn ici : leurs runtimes OpenMP entreraient en conflit avec celui de torch, utilisé
par d'autres tests du même processus pytest.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from features.features_eng import compute_hurst_exponent, compute_ou_mle
from research.backtest.engine import BacktestConfig, BacktestEngine, causal_max_hold
from research.institutional.feature_master import FeatureMaster
from research.institutional.locked_policy import locked_positions
from research.institutional.meta_labeling import (
    EMBARGO_MS, SCHEMA, MetaOracle, OracleMismatch, chain_fingerprint, entry_features, feature_matrix, make_chain,
    purged_walk_forward, select_threshold, trade_net_returns,
)
from research.institutional.signal_engine import InstitutionalSignalEngine

# Long (1 -> 3), short (3 -> 5), long (5 -> 7), puis short ouvert en 7 et jamais refermé.
X = [0, -1.2, 0, 1.2, 0, -1.2, 0, 1.2, 1.2]


def _signal(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    def c(v):
        return np.full(n, float(v))

    return SimpleNamespace(
        betas=c(1.0), alphas=c(0.0), allow_flip=True,
        d_star=c(-1.0), b_star=c(1.0), long_exit_level=c(1.0), short_exit_level=c(-1.0),
        entry_long=(x < -1.0).astype(float), entry_short=(x > 1.0).astype(float), exit_signal=np.zeros(n),
        theta_hat=c(0.5), sigma_hat=c(0.5), spreads=x, price_a=np.exp(x), price_b=np.ones(n),
    )


def _engine(**kw):
    return BacktestEngine(BacktestConfig(position_size=1.0, time_stop_mult=None, **kw))


# ---------------------------------------------------------------- filtre par trade

def test_filter_is_called_once_per_trade_and_removes_only_rejected_trades():
    calls = []
    res = _engine().run(_signal(X), lock_hedge=True, keep_trade=lambda t: calls.append(t["entry"]) or t["side"] == 1)
    assert calls == [1, 3, 5, 7]                                              # un appel par trade, pas par barre
    assert res["positions"].tolist() == [0, 1, 1, 0, 0, 1, 1, 0, 0]           # shorts retirés, longs intacts
    assert [t["kept"] for t in res["trades"]] == [True, False, True, False]


def test_rejecting_the_open_trade_flattens_the_last_bar_too():
    res = _engine().run(_signal(X), lock_hedge=True, keep_trade=lambda t: t["kind"] != "fin de série")
    assert res["positions"].tolist() == [0, 1, 1, -1, -1, 1, 1, 0, 0]


def test_filter_requires_the_locked_policy():
    with pytest.raises(ValueError):
        _engine().run(_signal(X), keep_trade=lambda t: True)


# ---------------------------------------------------------------- label

@pytest.mark.parametrize("k, entry", [(0, 1), (1, 3), (2, 5)])
def test_label_return_is_what_the_backtester_books(k, entry):
    bt = _engine()
    res = bt.run(_signal(X), lock_hedge=True, keep_trade=lambda t: t["entry"] == entry)
    booked = res["df"]["net_pnl"].sum() / bt.cfg.initial_capital              # seul trade gardé, notional = capital initial
    assert abs(trade_net_returns(res, bt.costs)[k] - booked) < 1e-9


# ---------------------------------------------------------------- arrêt temporel

def test_time_stop_is_read_at_entry_and_frozen():
    x = np.array([0.0, -1.2] + [-0.5] * 10)
    n = len(x)

    def run(cap):
        return locked_positions(_signal(x), x, np.zeros(n), max_hold=cap)[0].tolist()

    short_at_entry = np.full(n, 100.0)
    short_at_entry[1] = 3
    long_at_entry = np.full(n, 3.0)
    long_at_entry[1] = 100
    assert run(short_at_entry) == run(3) == [0, 1, 1, 1] + [0] * 8
    assert run(long_at_entry) == [0] + [1] * 11


def test_causal_max_hold_bounds():
    theta = [np.nan, 0.0, 1e-4, 0.1, 10.0]
    assert causal_max_hold(theta, 8.0).tolist() == [2688, 2688, 2688, 55, 20]


def test_causal_time_stop_requires_the_locked_policy():
    bt = BacktestEngine(BacktestConfig(time_stop_causal=True))
    with pytest.raises(ValueError):
        bt.run(_signal(X))


# ---------------------------------------------------------------- découpage, seuil, matrice

def test_purged_walk_forward_never_trains_on_overlapping_labels():
    h = 3_600_000
    t_entry = np.arange(0, 6000, 10) * h
    t_exit = t_entry + 300 * h
    splits = list(purged_walk_forward(t_entry, t_exit, n_splits=5))
    assert len(splits) == 5
    seen = np.concatenate([va for _, va in splits])
    assert len(np.unique(seen)) == len(seen)
    for tr, va in splits:
        assert t_exit[tr].max() < t_entry[va].min() - EMBARGO_MS


def test_threshold_rejects_everything_without_a_positive_total():
    p = np.linspace(0.0, 1.0, 400)
    net = np.where(p > 0.6, 0.01, -0.01)
    thr = select_threshold(p, net)
    assert 0.5 < thr < 0.7 and net[p >= thr].sum() > 0
    assert select_threshold(p, -np.abs(net)) == float("inf")


def test_missing_feature_raises_and_infinite_becomes_nan():
    df = pd.DataFrame({"a": [1.0, np.inf], "b": [2.0, 3.0]})
    M = feature_matrix(df, ["b", "a"])
    assert M[0].tolist() == [2.0, 1.0] and np.isnan(M[1, 1])                  # jamais 0 : 0 se lirait « retour immédiat »
    with pytest.raises(KeyError):
        feature_matrix(df, ["a", "absente"])


# ---------------------------------------------------------------- features

def test_entry_features_describe_the_traded_spread_and_ignore_the_future():
    rng = np.random.default_rng(1)
    n, i = 1200, 1000
    lb = np.cumsum(0.01 * rng.standard_normal(n)) + 3.0
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.95 * sp[t - 1] + 0.006 * rng.standard_normal()
    la = 1.5 * lb + 0.5 + sp
    la_other = la.copy()
    la_other[i + 1:] += rng.normal(0.0, 0.05, n - i - 1)                      # autre futur, même passé
    engine = InstitutionalSignalEngine(window=156, hedge_window=156, cost=0.002)
    fm = FeatureMaster(signature_level=2)

    def feats(log_a):
        sig = engine.generate(np.arange(n), np.exp(log_a), np.exp(lb))
        return sig, entry_features(fm, log_a, lb, sig, i)

    sig, f = feats(la)
    _, f_other = feats(la_other)
    assert abs(f["spread_at_entry"] - sig.spreads[i]) < 1e-9
    assert f.keys() == f_other.keys()
    np.testing.assert_allclose(list(f.values()), list(f_other.values()), rtol=1e-9, atol=1e-12)
    # les features viennent du spread à couverture d'entrée, pas du log-ratio (beta = 1)
    traded = la[i - 127:i + 1] - sig.betas[i] * lb[i - 127:i + 1] - sig.alphas[i]
    assert f["ou_mu"] == compute_ou_mle(traded)["mu"]
    assert f["hurst"] == compute_hurst_exponent(traded)


# ---------------------------------------------------------------- oracle

def _payload(**over):
    engine, cfg = make_chain()
    payload = {"schema": SCHEMA, "model": None, "features": ["hurst"], "platt": (1.0, 0.0), "threshold": 0.55,
               "chain": chain_fingerprint(engine, cfg), "train_end_ms": 1_672_531_200_000}
    payload.update(over)
    return payload


def _dump(tmp_path, payload):
    path = tmp_path / "oracle.pkl"
    path.write_bytes(pickle.dumps(payload))
    return str(path)


def test_oracle_matching_the_chain_is_loaded(tmp_path):
    engine, cfg = make_chain()
    oracle = MetaOracle.load(_dump(tmp_path, _payload()), engine, cfg, as_of_ms=1_672_531_200_000)
    assert oracle.threshold == 0.55


def test_stale_oracles_are_refused(tmp_path):
    engine, cfg = make_chain()
    with pytest.raises(FileNotFoundError):
        MetaOracle.load(str(tmp_path / "absent.pkl"), engine, cfg)
    with pytest.raises(OracleMismatch):                                       # format v2 / v3 : aucune empreinte
        MetaOracle.load(_dump(tmp_path, {"model": None, "features": []}), engine, cfg)
    with pytest.raises(OracleMismatch):                                       # autre moteur HJB
        MetaOracle.load(_dump(tmp_path, _payload()), *make_chain(cost=0.001))
    with pytest.raises(OracleMismatch):                                       # entraîné après le début de l'évaluation
        MetaOracle.load(_dump(tmp_path, _payload()), engine, cfg, as_of_ms=1_640_995_200_000)


# ---------------------------------------------------------------- portage (backtester réel)

def test_carry_reaches_the_capital():
    from research.institutional.backtester import apply_carry
    bt = _engine(fee_rate=0.0, slippage_bps=0.0)
    df = bt.run(_signal(X), lock_hedge=True, keep_trade=lambda t: t["entry"] == 1)["df"]
    n, cap0 = len(df), bt.cfg.initial_capital
    np.testing.assert_allclose(apply_carry(df, np.zeros(n), np.zeros(n), cap0)["capital"], df["capital"])
    paid = apply_carry(df, np.full(n, 1e-3), np.zeros(n), cap0)
    # long A tenu pendant les barres 2 et 3, notional = capital initial, funding de 10 pb par barre
    assert abs((df["capital"].iloc[-1] - paid["capital"].iloc[-1]) - 2 * 1e-3 * cap0) < 1e-6
