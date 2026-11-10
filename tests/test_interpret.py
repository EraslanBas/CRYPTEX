"""Loadings and effect decomposition against exact ground truth."""
import numpy as np
import pytest

from cryptex.interpret import (
    decompose_effects,
    loading_agreement,
    loadings_by_regression,
    loadings_per_context,
    top_genes,
)


@pytest.fixture
def linear_truth():
    rng = np.random.default_rng(0)
    n_p, n_c, d, g = 30, 4, 5, 60
    B = rng.normal(size=(d, g))
    mu = rng.normal(size=(n_p, d))
    ga = rng.normal(size=(n_c, d)) * 0.5
    labels_p = [f"P{i}" for i in range(n_p)]
    pert = np.repeat(labels_p, n_c)
    ctx = np.tile([f"C{j}" for j in range(n_c)], n_p)
    Z = np.array([mu[i] + ga[j] for i in range(n_p) for j in range(n_c)])
    genes = np.array([f"g{i}" for i in range(g)])
    return dict(B=B, mu=mu, ga=ga, Z=Z, X=Z @ B, pert=pert, ctx=ctx,
                genes=genes, labels_p=labels_p)


def test_regression_loadings_recover_linear_map(linear_truth):
    L = loadings_by_regression(
        linear_truth["Z"], linear_truth["X"], linear_truth["genes"]
    )
    assert np.abs(L.to_numpy() - linear_truth["B"]).max() < 1e-10


def test_effect_decomposition_is_exact(linear_truth):
    d = decompose_effects(linear_truth["Z"], linear_truth["pert"], linear_truth["ctx"])
    # additive truth => zero interaction
    assert np.abs(d.delta.to_numpy()).max() < 1e-10
    # and the parts reconstruct the input exactly
    recon = np.array([
        d.grand_mean + d.mu.loc[p].to_numpy() + d.gamma.loc[c].to_numpy()
        + d.delta.loc[(p, c)].to_numpy()
        for p, c in zip(linear_truth["pert"], linear_truth["ctx"])
    ])
    assert np.abs(recon - linear_truth["Z"]).max() < 1e-10


def test_effect_decomposition_main_effects_align(linear_truth):
    d = decompose_effects(linear_truth["Z"], linear_truth["pert"], linear_truth["ctx"])
    truth = linear_truth["mu"] - linear_truth["mu"].mean(0)
    order = [linear_truth["labels_p"].index(n) for n in d.mu.index]
    assert np.abs(d.mu.to_numpy() - truth[order]).max() < 1e-10


def test_interaction_is_detected_when_present(linear_truth):
    rng = np.random.default_rng(1)
    Z = linear_truth["Z"] + rng.normal(size=linear_truth["Z"].shape) * 0.8
    shares = decompose_effects(Z, linear_truth["pert"], linear_truth["ctx"]).variance_shares()
    assert shares["interaction"] > 0.1


def test_per_context_loadings_agree_for_a_global_map(linear_truth):
    pc = loadings_per_context(
        linear_truth["Z"], linear_truth["X"], linear_truth["ctx"], linear_truth["genes"]
    )
    assert loading_agreement(pc).to_numpy().min() > 0.99


def test_top_genes_splits_tails(linear_truth):
    L = loadings_by_regression(
        linear_truth["Z"], linear_truth["X"], linear_truth["genes"]
    )
    both = top_genes(L, 0, n=5, tail="both")
    assert len(both) == 10
    assert set(both["tail"]) == {"positive", "negative"}
