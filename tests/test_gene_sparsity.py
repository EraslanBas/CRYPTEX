"""Gene-side prior: each program concentrates its weight in few genes."""

from __future__ import annotations

import numpy as np
import torch

from cryptex.models.loadings import gene_concentration
from cryptex.train import TrainConfig, fit


def test_concentration_is_the_top_k_share():
    B = torch.tensor([[3.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(gene_concentration(B, 1), torch.tensor([16 / 25, 0.25]))
    assert torch.allclose(gene_concentration(B, 2), torch.tensor([1.0, 0.5]))
    assert torch.allclose(gene_concentration(5 * B, 2), gene_concentration(B, 2))   # scale-free


def test_gradient_moves_weight_into_the_top_genes():
    B = torch.tensor([[2.0, 1.0, 1.0, 1.0]], requires_grad=True)
    (1 - gene_concentration(B, 1)).sum().backward()
    assert B.grad[0, 0] < 0 and (B.grad[0, 1:] > 0).all()      # descent grows the top gene, shrinks the rest


def _data(seed=0):
    rng = np.random.default_rng(seed)
    G, d = 60, 3
    B = np.zeros((d, G)); B[0, :8] = 1; B[1, 8:16] = 1; B[2, 16:24] = 1   # programs on disjoint gene sets
    p = np.repeat(np.arange(45) // 3, 4); c = np.repeat(np.arange(45) % 3, 4)
    Z = rng.standard_normal((45, d)) * (rng.random((45, d)) < 0.5)
    X = (Z[np.repeat(np.arange(45), 4)] @ B + 0.05 * rng.standard_normal((len(p), G))).astype(np.float32)
    return X, p, c


def test_penalty_raises_gene_concentration_in_fit():
    X, p, c = _data()
    shares = {}
    for g in (0.0, 5.0):
        cfg = TrainConfig(pert_embedding_init="random", n_factors=3, epochs=40, batch_size=64, log_every=0, seed=0, subspace="fixed", subspace_method="pca", axes_init="random", noise_rank=2,
                          gene_sparsity=g, lr=1e-2)
        model, hist = fit(X, p, c, config=cfg)
        shares[g] = float(gene_concentration(model.loadings.B.detach(), 8).mean())
        assert "gene_top_share" in hist[-1]
    assert shares[5.0] > shares[0.0] + 0.05 and shares[5.0] > 0.9


def test_val_total_is_the_training_objective_term_by_term():
    X, p, c = _data()
    val = np.zeros(len(X), bool); val[::5] = True
    cfg = TrainConfig(pert_embedding_init="random", n_factors=3, epochs=3, batch_size=64, log_every=0, seed=0, subspace="fixed", subspace_method="pca", axes_init="random",
                      noise_rank=2, eval_every=1, coding="group_lasso", lasso_lambda=0.05, lasso_steps=10,
                      gene_sparsity=0.5)
    _, hist = fit(X, p, c, config=cfg, train_rows=~val, val_rows=val)
    h = hist[-1]
    # no synthetic rows: the objective's terms reduce to the plain validation terms
    assert np.isclose(h["val_disc_objective"], h["val_disc"])
    expect = (h["val_disc_objective"] + cfg.alpha * h["val_lasso_objective"] + 0.5 * h["val_gene_penalty"]
              + cfg.distinct_weight * h["val_distinct_penalty"])
    assert np.isclose(h["val_total"], expect)
    assert np.isclose(h["total"], h["disc"] + cfg.alpha * h["recon"] + 0.5 * h["gene_penalty"]
                      + cfg.distinct_weight * h["distinct_penalty"])


def test_l1_sparseness_and_collinearity_hinge():
    from cryptex.models.loadings import collinearity_hinge, l1_sparseness
    B = torch.tensor([[1.0, 0, 0, 0], [1.0, 1, 1, 1], [1.0, 0.1, 0, 0]])
    assert torch.allclose(l1_sparseness(B), torch.tensor([0.5, 1.0, 1.1 / (np.sqrt(1.01) * 2)], dtype=B.dtype))
    # rows 0 and 2 are nearly collinear (cos 0.995); rows 0 and 1 have cos 0.5
    assert float(collinearity_hinge(B, 0.99)) > 0 and float(collinearity_hinge(B, 0.999)) == 0.0
    assert np.isclose(float(collinearity_hinge(5 * B, 0.99)), float(collinearity_hinge(B, 0.99)), rtol=1e-4)


def test_hinge_stops_programs_collapsing_under_the_prior():
    X, p, c = _data()
    maxcos = {}
    for mu in (0.0, 100.0):
        cfg = TrainConfig(n_factors=3, epochs=40, batch_size=64, log_every=0, seed=0, subspace="fixed", subspace_method="pca", axes_init="random", noise_rank=2,
                          gene_sparsity=5.0, distinct_weight=mu, distinct_tau=0.5, lr=1e-2,
                          pert_embedding_init="random")
        model, hist = fit(X, p, c, config=cfg)
        Bu = model.loadings.B.detach(); Bu = Bu / Bu.norm(dim=1, keepdim=True)
        maxcos[mu] = float(((Bu @ Bu.T).abs() - torch.eye(3)).max())
        assert "distinct_penalty" in hist[-1]
    assert maxcos[100.0] <= 0.55
