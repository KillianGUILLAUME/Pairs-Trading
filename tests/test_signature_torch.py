import numpy as np
import pytest
import torch

from neural_nets.signature_torch import signature, sig_channels, TruncatedSignature

iisignature = pytest.importorskip("iisignature")


@pytest.mark.parametrize("d,L,depth", [(2, 20, 3), (3, 128, 3), (2, 7, 4), (3, 2, 2), (1, 5, 3)])
def test_matches_iisignature(d, L, depth):
    torch.manual_seed(0)
    p = torch.randn(4, L, d, dtype=torch.float64).cumsum(1)
    ours = signature(p, depth).numpy()
    ref = np.stack([iisignature.sig(p[i].numpy(), depth) for i in range(4)])
    assert ours.shape == (4, sig_channels(d, depth))
    np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-10)


def test_time_augmentation_matches_iisignature_on_augmented_path():
    torch.manual_seed(1)
    p = torch.randn(3, 50, 2, dtype=torch.float64).cumsum(1)
    ours = signature(p, 3, time_augment=True).numpy()
    t = np.linspace(0, 1, 50)[:, None]
    ref = np.stack([iisignature.sig(np.hstack([t, p[i].numpy()]), 3) for i in range(3)])
    np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-10)


def test_reparametrisation_invariance_only_without_time_channel():
    torch.manual_seed(2)
    p = torch.randn(2, 30, 2, dtype=torch.float64).cumsum(1)
    q = p.repeat_interleave(3, dim=1)  # même géométrie, 3x plus lent
    assert (signature(p, 3) - signature(q, 3)).abs().max() < 1e-9
    assert (signature(p, 3, time_augment=True) - signature(q, 3, time_augment=True)).abs().max() > 1e-2


def test_gradients_flow():
    p = torch.randn(2, 30, 2, requires_grad=True)
    s = TruncatedSignature(3, time_augment=True)(p)
    s.sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert s.shape[1] == TruncatedSignature(3, time_augment=True).out_channels(2)
