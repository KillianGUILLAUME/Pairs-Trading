import numpy as np
from types import SimpleNamespace

from research.institutional.locked_policy import locked_positions


def _signal(x):
    n = len(x)
    c = lambda v: np.full(n, float(v))
    return SimpleNamespace(
        betas=c(1.0), alphas=c(0.0), allow_flip=True,
        d_star=c(-1.0), b_star=c(1.0), long_exit_level=c(1.0), short_exit_level=c(-1.0),
        entry_long=(x < -1.0).astype(float), entry_short=(x > 1.0).astype(float),
        theta_hat=c(0.5), sigma_hat=c(0.5),           # sigma_inf = 0.5 / sqrt(1) = 0.5
    )


def _run(x, stop_sd):
    log_a, log_b = np.asarray(x, float), np.zeros(len(x))
    return locked_positions(_signal(np.asarray(x, float)), log_a, log_b, max_hold=10_000, stop_sd=stop_sd, return_trades=True)


# x : entrée long à -1.2, divergence à -1.9 (stop à -1 - 1*0.5 = -1.5), rentrée dans la bande à -0.5, nouvelle entrée à -1.2, retour à +1.1
X = [0, -1.2, -1.9, -1.9, -1.9, -0.5, -1.2, 0.0, 1.1, 1.1]


def test_stop_triggers_blocks_while_diverging_and_rearms():
    pos, _, trades = _run(X, stop_sd=1.0)
    # entrée (1), stop à la barre 2 -> plat, bloqué tant que x < -1 (barres 3-4), débloqué à -0.5, nouvelle entrée à -1.2
    assert pos.tolist() == [0, 1, 0, 0, 0, 0, 1, 1, -1, -1]
    kinds = [t["kind"] for t in trades]
    assert kinds[0] == "stop" and "retournement" in kinds
    assert trades[0]["mae_sd"] >= 1.0


def test_without_stop_the_position_is_held_through_the_divergence():
    pos, _, trades = _run(X, stop_sd=None)
    assert pos.tolist()[:6] == [0, 1, 1, 1, 1, 1]
    assert max(t["mae_sd"] for t in trades) >= 1.4          # excursion adverse mesurée même sans stop


def test_wide_stop_does_not_trigger():
    pos, _, trades = _run(X, stop_sd=3.0)
    assert all(t["kind"] != "stop" for t in trades)
