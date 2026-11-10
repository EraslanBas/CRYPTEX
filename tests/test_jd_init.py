"""Starting the axes at the joint-diagonalisation axes (axes_init='jd')."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from cryptex.train import TrainConfig, fit, jd_axes


def _context_variance_data(d=5, G=50, n_p=300, n_c=6, seed=0):
    """Pair means = s B_true, with each context scaling each source's variance."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, G)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    scale = rng.uniform(0.2, 3.0, size=(n_c, d))
    p = np.repeat(np.arange(n_p), n_c); c = np.tile(np.arange(n_c), n_p)
    S = rng.standard_normal((len(p), d)) * scale[c]
    X = (S @ B + 0.01 * rng.standard_normal((len(p), G))).astype(np.float32)
    V = np.linalg.svd(X, full_matrices=False)[2][:d]                 # span of B
    return X, p, c, B, V


def _match(B1, B2):
    C = np.abs((B1 / np.linalg.norm(B1, axis=1, keepdims=True)) @ (B2 / np.linalg.norm(B2, axis=1, keepdims=True)).T)
    r, k = linear_sum_assignment(-C)
    return C[r, k]


def test_jd_axes_recover_planted_programs():
    X, p, c, B, V = _context_variance_data(n_p=3000)
    A0 = jd_axes(X, p, c, np.arange(len(X)), V)
    B0 = A0 @ V
    assert np.allclose(np.linalg.norm(B0, axis=1), 1.0, atol=1e-5)
    assert _match(B0, B).min() > 0.99


def test_fit_starts_at_the_jd_axes():
    X, p, c, B, V = _context_variance_data(n_p=60)
    cfg = TrainConfig(pert_embedding_init="random", n_factors=5, epochs=0, batch_size=128, log_every=0, seed=0,
                      subspace="fixed", subspace_method="pca", axes_init="jd", noise_rank=2)
    model, _ = fit(X, p, c, config=cfg)
    expect = jd_axes(X, p, c, np.arange(len(X)), model.loadings.V.cpu().numpy())
    assert np.allclose(model.loadings.A.detach().cpu().numpy(), expect, atol=1e-5)

