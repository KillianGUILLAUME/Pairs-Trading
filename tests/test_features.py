import numpy as np

from features.features_eng import compute_hurst_exponent, compute_tail_risk, get_screener_feature_dict, SCREENER_FEATURE_KEYS


def _ou(rng, theta, n=128, sigma=0.01):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] + sigma * rng.standard_normal()
    return x


def test_hurst_orders_mean_reversion_random_walk_persistence():
    rng = np.random.default_rng(0)
    h_ou = np.mean([compute_hurst_exponent(_ou(rng, 0.1)) for _ in range(200)])
    h_rw = np.mean([compute_hurst_exponent(np.cumsum(rng.standard_normal(128))) for _ in range(200)])

    def persistent():
        inc = np.zeros(128)
        for t in range(1, 128):
            inc[t] = 0.6 * inc[t - 1] + rng.standard_normal()
        return np.cumsum(inc)

    h_pers = np.mean([compute_hurst_exponent(persistent()) for _ in range(200)])
    assert h_ou < 0.4
    assert 0.4 < h_rw < 0.55
    assert h_pers > 0.55
    assert h_ou < h_rw < h_pers


def test_hurst_rs_on_increments_not_levels():
    rng = np.random.default_rng(1)
    h = np.mean([compute_hurst_exponent(_ou(rng, 0.1), method="rs") for _ in range(100)])
    assert h < 0.7  # l'ancienne version (R/S sur les niveaux) donnait ~0.9


def test_fast_tail_estimator_is_close_to_mle():
    rng = np.random.default_rng(2)
    x = rng.standard_t(4, 20000) * 0.01
    assert abs(compute_tail_risk(x, fast=True) - 4.0) < 1.0


def test_feature_dict_keys_and_speed():
    rng = np.random.default_rng(3)
    la = np.cumsum(0.01 * rng.standard_normal(128))
    lb = la - _ou(rng, 0.1)
    lbtc = np.cumsum(0.01 * rng.standard_normal(128))
    d = get_screener_feature_dict(la, lb, lbtc, lbtc + 0.1, fast=True)
    assert list(d.keys()) == SCREENER_FEATURE_KEYS
    assert all(np.isfinite(v) or k == "half_life" for k, v in d.items())
