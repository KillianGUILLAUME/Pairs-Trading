import os

import numpy as np
import pytest
import torch
from scipy import stats

from neural_nets.neural_sde import GeneratorSDE, _inv_softplus

TS = torch.linspace(0.0, 127.0, 128)


def _model(**kw):
    torch.manual_seed(0)
    return GeneratorSDE(data_dim=2, hidden_dim=16, condition_dim=3, dt=0.5, **kw)


def _gen(m, B=400, cond=None, seed=0):
    torch.manual_seed(seed)
    c = torch.zeros(B, 3) if cond is None else cond
    with torch.no_grad():
        return m(torch.zeros(B, 2), TS, c).numpy()


def _incr_stats(paths):
    d = np.diff(paths[:, :, 0], axis=1)
    sq = d ** 2
    acf = np.mean([np.corrcoef(sq[i, 1:], sq[i, :-1])[0, 1] for i in range(len(sq))])
    return float(stats.kurtosis(d.ravel())), float(acf)


def test_extended_model_shapes_gradients_and_finiteness():
    m = _model(latent_dim=1, noise="student", mean_reversion=True)
    m.train()
    y0 = torch.zeros(8, 2)
    out = m(y0, TS[:20], torch.randn(8, 3))
    assert out.shape == (8, 20, 2) and torch.isfinite(out).all()
    assert m.last_latent.shape == (8, 20, 1)
    # le gradient atteint tous les paramètres, dont les nouveaux
    loss = (out ** 2).mean() + out[:, -1].abs().mean()
    loss.backward()
    names = ["kappa_raw", "xi_raw", "v0_scale_raw", "nu_raw", "theta_bias", "mu_bias"]
    for n in names:
        g = getattr(m, n).grad
        assert g is not None and torch.isfinite(g).all(), n
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.theta_head.parameters())


def test_default_model_keeps_legacy_torchsde_path():
    m = _model()
    assert m.custom is False and not hasattr(m, "kappa_raw")
    out = _gen(m, B=16)
    assert out.shape == (16, 128, 2) and np.isfinite(out).all()


def test_latent_volatility_creates_clustering_and_fat_tails():
    base = _model(latent_dim=1)
    # volatilité de volatilité nulle : contrôle sans clustering
    with torch.no_grad():
        base.xi_raw.fill_(_inv_softplus(1e-4)); base.v0_scale_raw.fill_(_inv_softplus(1e-4))
    k0, a0 = _incr_stats(_gen(base))
    sv = _model(latent_dim=1)
    with torch.no_grad():
        sv.xi_raw.fill_(_inv_softplus(0.25)); sv.kappa_raw.fill_(_inv_softplus(0.02)); sv.v0_scale_raw.fill_(_inv_softplus(0.5))
    k1, a1 = _incr_stats(_gen(sv))
    assert abs(k0) < 0.5 and abs(a0) < 0.05              # gaussien, pas de clustering
    assert a1 > 0.05 and k1 > 1.0                          # clustering et queues épaisses


def test_student_noise_creates_fat_tails_without_clustering():
    m = _model(noise="student", student_nu=4.5)
    k, a = _incr_stats(_gen(m))
    assert k > 2.0 and abs(a) < 0.05
    g = _model(noise="gaussian", mean_reversion=True)     # contrôle (chemin maison, bruit gaussien)
    kg, _ = _incr_stats(_gen(g))
    assert abs(kg) < 0.5


def test_student_noise_has_unit_variance_scale():
    m = _model(noise="student", student_nu=4.5)
    z = m._increments(200000, 3, torch.device("cpu"), torch.float32)[:, :2]
    assert abs(float(z.std()) - 1.0) < 0.05


def test_mean_reversion_gives_stationary_spread_and_conditioning_matters():
    def ar1_theta(paths):
        s = paths[:, :, 0] - paths[:, :, 1]
        b = np.mean([np.polyfit(s[i, :-1], s[i, 1:], 1)[0] for i in range(len(s))])
        return -np.log(b)

    plain = _model(noise="gaussian", mean_reversion=True)
    with torch.no_grad():
        plain.theta_bias.fill_(_inv_softplus(1e-3))
    mr = _model(noise="gaussian", mean_reversion=True)
    with torch.no_grad():
        mr.theta_bias.fill_(_inv_softplus(0.15))
    assert ar1_theta(_gen(mr)) > ar1_theta(_gen(plain)) + 0.05
    # le conditionnement agit : θ dépend de c
    cm = _model(noise="gaussian", mean_reversion=True)
    with torch.no_grad():
        cm.theta_head.weight.copy_(torch.tensor([[3.0, 0.0, 0.0]]))
        cm.theta_bias.fill_(_inv_softplus(0.05))
    hi = _gen(cm, cond=torch.tensor([[1.5, 0.0, 0.0]]).repeat(400, 1))
    lo = _gen(cm, cond=torch.tensor([[-1.5, 0.0, 0.0]]).repeat(400, 1))
    s_hi = (hi[:, :, 0] - hi[:, :, 1]).std(); s_lo = (lo[:, :, 0] - lo[:, :, 1]).std()
    assert ar1_theta(hi) > ar1_theta(lo) + 0.05 and s_hi != s_lo


def test_checkpoint_roundtrip_preserves_architecture_flags(tmp_path):
    m = _model(latent_dim=1, noise="student", mean_reversion=True)
    cfg = {"model": {"data_dim": 2, "latent_dim": 1, "noise": "student", "mean_reversion": True, "dt_sde": 0.5, "student_nu": 6.0}}
    p = tmp_path / "ck.pt"
    torch.save({"model_state_dict": m.state_dict(), "config": cfg}, p)
    m2, _ = GeneratorSDE.load_from_checkpoint(str(p))
    assert m2.latent_dim == 1 and m2.noise == "student" and m2.mean_reversion and m2.condition_dim == 3
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


@pytest.mark.skipif(not os.path.exists("data/models/neural_sde_20260421_001939_world_model.pt"), reason="checkpoint d'avril absent")
def test_april_checkpoint_still_loads_and_samples():
    m, ck = GeneratorSDE.load_from_checkpoint("data/models/neural_sde_20260421_001939_world_model.pt")
    assert m.custom is False
    out = m.sample(torch.zeros(4, 2), 32, torch.zeros(4, m.condition_dim))
    assert out.shape == (4, 32, 2) and torch.isfinite(out).all()
