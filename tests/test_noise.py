"""Noise model, label-driven subspace, and the fixed / anchored subspace modes."""

import numpy as np
import pandas as pd
import pytest
import torch

from cryptex.data.noise import (NoiseModel, label_subspace, noise_covariance, pair_means,
                                pca_subspace, rca_subspace, within_pair_residuals)
from cryptex.models import FixedBasisLoadings, GlobalLoadings


def _planted(n_pairs=600, G=60, d=3, r=2, n_rep=7, seed=0):
    """Pairs of exchangeable rows x = z A + noise, the noise with a planted
    correlated part (covariance diag + U diag(s) U^T, the same for every row)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, G))
    U, _ = np.linalg.qr(rng.normal(size=(G, r)))
    s_true = np.array([4.0, 2.0])
    diag = rng.uniform(0.5, 1.5, G)
    L = U * np.sqrt(s_true)
    rows, X = [], []
    for p in range(n_pairs):
        z = rng.laplace(size=d) * 0.5
        for k in range(n_rep):
            X.append(z @ A + rng.normal(size=G) * np.sqrt(diag) + rng.normal(size=r) @ L.T)
            rows.append(dict(perturbation=f"P{p}", context="C0", is_real=True))
    return np.asarray(X, np.float32), pd.DataFrame(rows), A, U, s_true, diag


def _pair(meta):
    return (meta.perturbation + "|" + meta.context).to_numpy()


def test_within_pair_residuals_recover_the_noise():
    X, meta, A, U, s_true, diag = _planted()
    R = within_pair_residuals(X, _pair(meta), np.ones(len(meta), bool))
    assert len(R) == len(meta)
    nm = noise_covariance(R, rank=2)
    # per-gene total noise variance and the correlated directions are recovered
    total_true = diag + (U * U * s_true).sum(1)
    total_hat = nm.diag + (nm.U * nm.U * nm.s).sum(1)
    assert np.corrcoef(total_true, total_hat)[0, 1] > 0.9
    cos = np.abs(nm.U.T @ U)
    assert np.linalg.svd(cos, compute_uv=False).min() > 0.9
    assert np.allclose(np.sort(nm.s)[::-1], s_true, rtol=0.25)


def test_pairs_with_one_row_give_no_residual():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    R = within_pair_residuals(X, np.array(["a", "a", "b", "c"]), np.ones(4, bool))
    assert R.shape == (2, 3)
    assert np.allclose(R[0], -R[1])


def test_metric_square_roots_are_consistent():
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    nm = NoiseModel(diag=rng.uniform(0.5, 2, 20), U=U, s=np.array([5.0, 2.0, 1.0]))
    M = nm.c * np.eye(20) + U @ np.diag(nm.s) @ U.T
    X = rng.normal(size=(4, 20))
    assert np.allclose(nm.metric_sqrt(nm.metric_inv_sqrt(X)), X)
    assert np.allclose(nm.metric_inv_quadratic(X), np.einsum("ij,jk,ik->i", X, np.linalg.inv(M), X))


def test_label_subspace_finds_the_signal_span():
    X, meta, A, *_ = _planted()
    allr = np.ones(len(meta), bool)
    R = within_pair_residuals(X, _pair(meta), allr)
    _, means, counts = pair_means(X, _pair(meta), allr)
    V, ev = label_subspace(means, counts, noise_covariance(R, 2), d=3)
    assert V.shape == (3, X.shape[1]) and np.all(np.diff(ev) <= 0)
    Qa, _ = np.linalg.qr(A.T); Qv, _ = np.linalg.qr(V.T)
    assert np.linalg.svd(Qa.T @ Qv, compute_uv=False).min() > 0.95


def test_fixed_basis_keeps_the_span():
    rng = np.random.default_rng(2)
    V = rng.normal(size=(4, 30)).astype(np.float32)
    fl = FixedBasisLoadings(V)
    assert np.allclose(fl.B.detach().numpy(), V)
    with torch.no_grad():
        fl.A.copy_(torch.as_tensor(rng.normal(size=(4, 4)), dtype=torch.float32))
    x = torch.as_tensor(rng.normal(size=(8, 30)), dtype=torch.float32)
    gl = GlobalLoadings(4, 30)
    with torch.no_grad():
        gl.B.copy_(torch.as_tensor(V))
    # same span -> same reconstruction, whatever A is
    assert torch.allclose(fl.reconstruct(fl.project(x)), gl.reconstruct(gl.project(x)), atol=1e-4)
    assert [n for n, _ in fl.named_parameters()] == ["A"]


def test_fit_runs_in_each_subspace_mode():
    from cryptex.train import TrainConfig, fit
    X, meta, *_ = _planted(n_pairs=80, G=30, n_rep=4)
    meta["context"] = np.where(np.arange(len(meta)) % 2, "C0", "C1")
    p = pd.factorize(meta.perturbation)[0]; c = pd.factorize(meta.context)[0]
    allr = np.ones(len(meta), bool)
    nm = noise_covariance(within_pair_residuals(X, _pair(meta), allr), rank=2)
    _, means, counts = pair_means(X, _pair(meta), allr)
    V, _ = label_subspace(means, counts, nm, d=3)
    val = meta.perturbation.isin([f"P{i}" for i in range(0, 80, 9)]).to_numpy()
    for mode in ("free", "fixed"):
        cfg = TrainConfig(pert_embedding_init="random", n_factors=3, epochs=2, batch_size=64, eval_every=1,
                          log_every=0, subspace=mode)
        model, hist = fit(X, p, c, config=cfg, train_rows=~val, val_rows=val, subspace_basis=V)
        assert "val_auc" in hist[-1]
        if mode == "fixed":
            B = model.loading_matrix()
            Qb, _ = np.linalg.qr(B.T); Qv, _ = np.linalg.qr(V.T)
            assert np.linalg.svd(Qb.T @ Qv, compute_uv=False).min() > 0.999
    # an unknown mode is refused; fixed computes V itself from the training rows
    with pytest.raises(ValueError):
        fit(X, p, c, config=TrainConfig(pert_embedding_init="random", n_factors=3, subspace="anchored"), train_rows=~val)
    cfg = TrainConfig(pert_embedding_init="random", n_factors=3, epochs=1, batch_size=64, eval_every=0,
                      log_every=0, subspace="fixed", noise_rank=2)
    model, _ = fit(X, p, c, config=cfg, train_rows=~val)
    assert model.subspace_basis_.shape == (3, X.shape[1])
    Qb, _ = np.linalg.qr(model.loading_matrix().T); Qv, _ = np.linalg.qr(model.subspace_basis_.T)
    assert np.linalg.svd(Qb.T @ Qv, compute_uv=False).min() > 0.999


def test_random_axes_init_differs_by_seed_and_keeps_the_span():
    rng = np.random.default_rng(4)
    V = rng.normal(size=(4, 30)).astype(np.float32)
    a = FixedBasisLoadings(V, axes_init="random", seed=0)
    b = FixedBasisLoadings(V, axes_init="random", seed=1)
    assert not torch.allclose(a.A, b.A)
    A = a.A.detach().numpy()
    assert np.allclose(A @ A.T, np.eye(4), atol=1e-5)          # a rotation
    Qb, _ = np.linalg.qr(a.B.detach().numpy().T); Qv, _ = np.linalg.qr(V.T)
    assert np.linalg.svd(Qb.T @ Qv, compute_uv=False).min() > 0.999
    assert torch.allclose(FixedBasisLoadings(V).A, torch.eye(4))  # default unchanged


def test_rca_matches_brute_force_and_finds_the_signal_span():
    X, meta, A, *_ = _planted(n_pairs=200, G=30, n_rep=4)
    pair = _pair(meta)
    G = X.shape[1]; Rb = np.zeros((G, G)); Rw = np.zeros((G, G))
    for key in np.unique(pair):
        Xi = X[pair == key].astype(float); k = len(Xi)
        Rw += (k - 1) * Xi.T @ Xi
        tot = Xi.sum(0)
        Rb += np.outer(tot, tot) - Xi.T @ Xi
    from scipy.linalg import eigh
    rho, W = eigh(Rb, Rw)
    V, r = rca_subspace(X, pair, np.ones(len(meta), bool), d=3, rank=G)
    assert np.allclose(r, rho[::-1][:3], atol=1e-4)
    Vb = (Rw @ W[:, ::-1][:, :3]).T
    ov = lambda a, b: np.linalg.svd(np.linalg.qr(a.T)[0].T @ np.linalg.qr(b.T)[0], compute_uv=False).min()
    assert ov(V, Vb) > 0.999 and ov(V, A) > 0.99
    assert np.allclose(np.linalg.norm(V, axis=1), 1)


def test_pca_subspace_is_the_top_svd_of_the_pair_means():
    X, meta, A, *_ = _planted(n_pairs=200, G=30, n_rep=4)
    allr = np.ones(len(meta), bool)
    V, _ = pca_subspace(X, _pair(meta), allr, d=3)
    _, means, _ = pair_means(X, _pair(meta), allr)
    top = np.linalg.svd(means, full_matrices=False)[2][:3]
    ov = np.linalg.svd(np.linalg.qr(V.T)[0].T @ np.linalg.qr(top.T)[0], compute_uv=False).min()
    assert ov > 0.999


def test_free_mode_can_start_where_the_fixed_mode_starts():
    from cryptex.train import Cryptex, TrainConfig
    V = np.random.default_rng(0).normal(size=(4, 30)).astype(np.float32)
    kw = dict(n_genes=30, n_perturbations=5, n_contexts=2, subspace_basis=V)
    fixed = Cryptex(config=TrainConfig(pert_embedding_init="random", n_factors=4, subspace="fixed", axes_init="random", seed=3), **kw)
    free = Cryptex(config=TrainConfig(pert_embedding_init="random", n_factors=4, subspace="free", free_init="basis",
                                      axes_init="random", seed=3), **kw)
    assert torch.allclose(fixed.loadings.B, free.loadings.B, atol=1e-6)
    assert "loadings.B" in dict(free.named_parameters())
