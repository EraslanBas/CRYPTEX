"""The statistics head with a learned label-only normaliser c_k(u)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cryptex.models import PerComponentHead
from cryptex.train import TrainConfig, fit


def _lam(zhat, c=None):
    """Coefficients giving the Gaussian score zhat z - z^2/2 (linear, square, abs, tanh)."""
    lam = torch.zeros(*zhat.shape, 4 if c is None else 5)
    lam[..., 0] = zhat; lam[..., 1] = -0.5
    if c is not None:
        lam[..., 4] = c
    return lam


def test_normaliser_recovers_the_distance_ranking():
    # two programs; label A centred at (2, 0), label B at (0, 4); the row (2, 1.5) is near A
    z = torch.tensor([[2.0, 1.5]])
    zA, zB = torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 4.0]])
    plain, normed = PerComponentHead(), PerComponentHead(normaliser=True)
    assert plain(z, _lam(zA)) < plain(z, _lam(zB))                          # without it B wins
    sA, sB = normed(z, _lam(zA, zA ** 2 / 2)), normed(z, _lam(zB, zB ** 2 / 2))
    assert sA > sB
    assert torch.allclose(sA, -((z - zA) ** 2).sum() / 2) and torch.allclose(sB, -((z - zB) ** 2).sum() / 2)


def test_default_head_is_unchanged_and_shapes_are_checked():
    z, lam = torch.randn(6, 3), torch.randn(6, 3, 4)
    h = PerComponentHead()
    assert h.n_outputs == 4 and torch.allclose(h(z, lam), (lam * h.statistics(z)).sum((1, 2)) + h.bias)
    with pytest.raises(ValueError):
        PerComponentHead(normaliser=True)(z, lam)


def test_fit_runs_with_the_normaliser_jd_start_and_group_lasso():
    rng = np.random.default_rng(0)
    p = np.repeat(np.arange(36) // 3, 4); c = np.repeat(np.arange(36) % 3, 4)
    B = rng.standard_normal((4, 40))
    X = (rng.standard_normal((36, 4))[np.repeat(np.arange(36), 4)] @ B
         + 0.1 * rng.standard_normal((len(p), 40))).astype(np.float32)
    val = np.zeros(len(X), bool); val[::5] = True
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=6, batch_size=50, log_every=0, seed=0, subspace="fixed", subspace_method="pca", axes_init="jd",
                      noise_rank=2, eval_every=3, coding="group_lasso", lasso_lambda=0.05, lasso_steps=10)
    model, history = fit(X, p, c, config=cfg, train_rows=~val, val_rows=val)
    assert model.head.normaliser and model.label_net.n_basis == 5
    assert history[-1]["disc"] < history[0]["disc"]

