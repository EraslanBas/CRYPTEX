"""Group-lasso activities: one selection of programs per (perturbation, context) pair."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cryptex.models.group_lasso import (
    group_batches, group_keys, group_lasso_codes, group_penalty)
from cryptex.train import TrainConfig, evaluate, fit, encode


def _planted(n_pairs=40, rows=6, d=8, G=60, k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(d, G, generator=g)
    B = B / B.norm(dim=1, keepdim=True)
    groups, X, support = [], [], []
    for i in range(n_pairs):
        s = torch.randperm(d, generator=g)[:k]
        z = torch.zeros(rows, d)
        z[:, s] = 1.0 + torch.rand(rows, k, generator=g)
        X.append(z @ B + 0.02 * torch.randn(rows, G, generator=g))
        groups += [i] * rows
        support.append(set(s.tolist()))
    return torch.cat(X), B, torch.tensor(groups), support


def test_selection_is_shared_within_a_group_and_recovers_the_support():
    X, B, groups, support = _planted()
    Z, sel = group_lasso_codes(X, B, groups, lam=0.3, steps=200)
    for i, s in enumerate(support):
        rows = sel[groups == i]
        assert (rows == rows[0]).all()
        assert set(torch.nonzero(rows[0]).flatten().tolist()) == s
    assert (Z[~sel] == 0).all()


def test_masked_gene_is_ignored_by_the_solver():
    X, B, groups, _ = _planted(n_pairs=2)
    cols = torch.full((len(X),), 5)
    X2 = X.clone(); X2[:, 5] += 100.0
    Z1, _ = group_lasso_codes(X, B, groups, 0.3, target_col=cols, steps=50)
    Z2, _ = group_lasso_codes(X2, B, groups, 0.3, target_col=cols, steps=50)
    assert torch.allclose(Z1, Z2, atol=1e-5)


def test_penalty_is_sqrt_group_size_times_column_norms():
    Z = torch.tensor([[3.0, 0.0], [4.0, 1.0], [0.0, 2.0]])
    groups = torch.tensor([7, 7, 2])
    expect = 0.5 * (np.sqrt(2) * (5.0 + 1.0) + 1.0 * (0.0 + 2.0))
    assert float(group_penalty(Z, groups, 0.5)) == pytest.approx(expect)


def test_gradients_reach_B():
    X, B, groups, _ = _planted(n_pairs=4)
    B = B.clone().requires_grad_(True)
    Z, _ = group_lasso_codes(X, B, groups, 0.3, steps=20)
    ((X - Z @ B) ** 2).sum().backward()
    assert B.grad is not None and float(B.grad.abs().sum()) > 0


def test_batches_hold_whole_groups():
    pair = np.repeat(np.arange(20), 7)                   # 20 pairs, 7 rows each
    keys = group_keys(pair % 5, pair // 5)
    for rng in (None, np.random.default_rng(0)):
        batches = group_batches(keys, 16, rng)
        assert sorted(np.concatenate(batches)) == list(range(len(keys)))
        seen = [set(keys[b]) for b in batches]
        assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1:])
    # one group per pair
    assert len(np.unique(keys)) == 20


def _toy(n_p=12, n_c=3, rows=4, d=4, G=40, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, G))
    eff = rng.standard_normal((n_p, n_c, d)) * (rng.random((n_p, n_c, d)) < 0.4)
    p = np.repeat(np.arange(n_p * n_c) // n_c, rows); c = np.repeat(np.arange(n_p * n_c) % n_c, rows)
    X = (eff[p, c] @ B + 0.1 * rng.standard_normal((len(p), G))).astype(np.float32)
    return X, p, c


@pytest.mark.parametrize("subspace", ["fixed", "free"])
def test_fit_and_evaluate_with_the_group_lasso(subspace):
    X, p, c = _toy()
    val = np.zeros(len(X), bool); val[::5] = True
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=6, batch_size=40, log_every=0, seed=0,
                      subspace=subspace, axes_init="random", noise_rank=2, eval_every=3,
                      coding="group_lasso", lasso_lambda=0.05, lasso_steps=10, lasso_groups="pair")
    model, history = fit(X, p, c, config=cfg, train_rows=~val, val_rows=val)
    assert np.allclose(model.loadings.B.detach().norm(dim=1).cpu().numpy(), 1.0, atol=1e-5)
    assert all(h["lasso_penalty"] > 0 and 0 < h["programs_per_pair"] <= 4 for h in history)
    val_rec = [h for h in history if "val_programs_per_pair" in h or "val_lasso_objective" in h]
    assert val_rec, history[-1].keys()
    with pytest.raises(ValueError):
        encode(model, X)
    Z = encode(model, X, groups=group_keys(p, c))
    for g in np.unique(group_keys(p, c)):
        nz = Z[group_keys(p, c) == g] != 0
        assert (nz == nz[0]).all() or not nz.any()


def test_row_lasso_activities_depend_on_the_row_alone():
    """With lasso_groups="row" a row's activities do not change when the other
    rows of its pair change or are dropped."""
    from cryptex.models.group_lasso import lasso_groups
    X, B, groups, _ = _planted(n_pairs=4)
    p = groups.numpy()                      # the pair ids
    g = torch.as_tensor(lasso_groups(p, np.zeros_like(p), "row"))
    Z_all, _ = group_lasso_codes(X, B, g, 0.3, steps=200)
    keep = torch.arange(0, len(X), 2)
    Z_half, _ = group_lasso_codes(X[keep], B, g[keep], 0.3, steps=200)
    assert torch.allclose(Z_all[keep], Z_half, atol=1e-5)


def test_fit_with_the_row_lasso():
    X, p, c = _toy()
    val = np.zeros(len(X), bool); val[::5] = True
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=4, batch_size=40, log_every=0, seed=0,
                      axes_init="random", eval_every=2, lasso_lambda=0.05, lasso_steps=10, lasso_groups="row")
    model, history = fit(X, p, c, config=cfg, train_rows=~val, val_rows=val)
    assert all(h["lasso_penalty"] > 0 for h in history)
    from cryptex.models.group_lasso import lasso_groups
    Z = encode(model, X, groups=lasso_groups(p, c, "row"))
    assert np.array_equal(Z, encode(model, X))        # per row: no groups needed
    nz = (Z != 0).sum(1)
    assert len(set(map(tuple, (Z != 0).astype(int)))) > 1 and nz.max() <= 4
