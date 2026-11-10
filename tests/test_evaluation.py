"""Matching, splits and scores."""
import numpy as np
import pytest

from cryptex.evaluation import (
    match_factors,
    reconstruction_score,
    split_context_holdout,
    split_pair_holdout,
    stability_across_seeds,
)


def test_match_is_invariant_to_permutation_and_sign():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 80))
    perm = rng.permutation(6)
    signs = rng.choice([-1, 1], 6)
    B = A[perm] * signs[:, None]
    m = match_factors(A, B)
    assert m.mcc > 0.999
    assert (m.permutation == np.argsort(perm)).all()


def test_stability_needs_two_runs():
    with pytest.raises(ValueError, match="at least two"):
        stability_across_seeds([np.zeros((3, 10))])


def test_pair_split_has_no_leakage():
    pert = np.repeat([f"P{i}" for i in range(20)], 3)
    ctx = np.tile(["A", "B", "C"], 20)
    s = split_pair_holdout(pert, ctx, frac=0.2, seed=1)
    train = set(zip(pert[s.train], ctx[s.train]))
    test = set(zip(pert[s.test], ctx[s.test]))
    assert not (train & test)
    assert s.train.sum() + s.test.sum() == len(pert)


def test_context_split_holds_out_whole_context():
    ctx = np.tile(["A", "B", "C"], 10)
    s = split_context_holdout(ctx, "B")
    assert set(ctx[s.test]) == {"B"}
    assert "B" not in set(ctx[s.train])


def test_unknown_context_rejected():
    with pytest.raises(ValueError, match="not present"):
        split_context_holdout(np.array(["A", "B"]), "Z")


def test_reconstruction_score_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 50))
    assert reconstruction_score(X, X)["r2"] == pytest.approx(1.0)
    assert reconstruction_score(X, rng.normal(size=(30, 50)))["r2"] < 0.1


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        reconstruction_score(np.zeros((3, 4)), np.zeros((3, 5)))
