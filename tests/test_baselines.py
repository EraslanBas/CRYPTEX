"""Baselines must recover a known linear mixing; PCA must not."""
import numpy as np
import pytest
from scipy.stats import ortho_group

import baselines
from cryptex.evaluation import match_factors


@pytest.fixture
def mixed_data():
    rng = np.random.default_rng(0)
    n_p, n_c, d, g = 60, 8, 6, 120
    B = rng.normal(size=(d, g))
    ctx = np.tile([f"C{j}" for j in range(n_c)], n_p)
    pert = np.repeat([f"P{i}" for i in range(n_p)], n_c)
    scale = rng.uniform(0.4, 2.5, size=(n_c, d))
    Z = rng.laplace(size=(n_p * n_c, d)) * np.array(
        [scale[j] for _ in range(n_p) for j in range(n_c)]
    )
    X = Z @ B + rng.normal(size=(n_p * n_c, g)) * 0.05
    return X, pert, ctx, B, d


def test_registry_lists_implemented_baselines():
    for name in ("pca", "linear_ica", "context_jd"):
        assert name in baselines.available()


def test_fastica_recovers_mixing(mixed_data):
    X, pert, ctx, B, d = mixed_data
    m = baselines.get("linear_ica")(n_factors=d).fit(X, pert, ctx)
    assert match_factors(B, m.loadings).mcc > 0.9


def test_context_jd_recovers_mixing(mixed_data):
    X, pert, ctx, B, d = mixed_data
    m = baselines.get("context_jd")(n_factors=d).fit(X, pert, ctx)
    assert match_factors(B, m.loadings).mcc > 0.9
    assert m.residual_off_diagonal_ < 0.2


def test_pca_does_not_unmix(mixed_data):
    """PCA is the honest floor: it spans the right subspace but its axes are an
    arbitrary rotation within it."""
    X, pert, ctx, B, d = mixed_data
    m = baselines.get("pca")(n_factors=d).fit(X)
    ica = baselines.get("linear_ica")(n_factors=d).fit(X)
    assert match_factors(B, m.loadings).mcc < match_factors(B, ica.loadings).mcc


def test_context_jd_needs_contexts(mixed_data):
    X, *_ = mixed_data
    with pytest.raises(ValueError, match="requires"):
        baselines.get("context_jd")().fit(X)


def test_joint_diagonalization_exact():
    rng = np.random.default_rng(0)
    d, K = 8, 16
    Q = ortho_group.rvs(d, random_state=1)
    M = np.stack([Q @ np.diag(rng.uniform(0.5, 4.0, d)) @ Q.T for _ in range(K)])
    V, _ = baselines.joint_diagonalize(M)
    assert baselines.off_diagonal_energy(M, V) < 1e-12
    assert match_factors(Q.T, V.T).mcc > 0.999
