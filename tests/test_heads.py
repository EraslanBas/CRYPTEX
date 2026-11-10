"""The per-component head must stay separable, and reject a degenerate basis."""
import torch

import pytest

from cryptex.models import FactorizedLabelNet, PerComponentHead


def test_purely_quadratic_basis_rejected():
    with pytest.raises(ValueError, match="rotation invariant"):
        PerComponentHead(basis=("square",))


def test_unknown_basis_rejected():
    with pytest.raises(ValueError, match="unknown basis"):
        PerComponentHead(basis=("cube",))


def test_score_is_additive_across_components():
    """score(z) must equal the sum of per-component scores -- i.e. no term
    couples z_k to z_j. This is the identifiability constraint; if it ever
    breaks, the factors stop being identified."""
    torch.manual_seed(0)
    head = PerComponentHead()
    d = 6
    z = torch.randn(4, d)
    lam = torch.randn(4, d, head.n_basis)

    total = head(z, lam) - head.bias
    per_component = torch.zeros_like(total)
    for k in range(d):
        z_k = torch.zeros_like(z)
        z_k[:, k] = z[:, k]
        lam_k = torch.zeros_like(lam)
        lam_k[:, k] = lam[:, k]
        per_component += head(z_k, lam_k) - head.bias
    assert torch.allclose(total, per_component, atol=1e-5)


def test_shape_mismatch_is_caught():
    head = PerComponentHead()
    with pytest.raises(ValueError, match="expected"):
        head(torch.randn(4, 5), torch.randn(4, 5, head.n_basis + 1))


def test_label_net_shape():
    head = PerComponentHead()
    net = FactorizedLabelNet(12, 3, 7, head.n_basis)
    lam = net(torch.randint(0, 12, (5,)), torch.randint(0, 3, (5,)))
    assert lam.shape == (5, 7, head.n_basis)
