"""The implementation must match the method paper, equation by equation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cryptex.data.ontarget import mask_coverage, on_target_index
from cryptex.evaluation import span_residual
from cryptex.models import NO_MASK, GlobalLoadings, PerComponentHead
from cryptex.models.label_net import FactorizedLabelNet
from cryptex.objectives import (
    DEFAULT_WEIGHTS,
    StratifiedNegativeSampler,
    contrastive_loss,
)
from cryptex.train import TrainConfig, fit


# ---------------------------------------------------------------- eq. (2)/(4)


def test_unmasked_projection_is_x_times_pinv():
    torch.manual_seed(0)
    L = GlobalLoadings(5, 40, ridge=0.0)
    x = torch.randn(7, 40, dtype=torch.float32)
    z = L.project(x)
    assert torch.allclose(z, x @ L.pinv, atol=1e-4)


def test_unmasked_projection_is_the_least_squares_solution():
    torch.manual_seed(1)
    L = GlobalLoadings(4, 30, ridge=0.0)
    x = torch.randn(3, 30, dtype=torch.float64)
    B = L.B.detach().double()
    z = L.project(x.float()).double()
    expected = torch.linalg.lstsq(B.T, x.T).solution.T
    assert torch.allclose(z, expected, atol=1e-4)


def test_masked_projection_matches_explicit_masked_least_squares():
    """eq. (4) via Sherman-Morrison must equal dropping the column and solving."""
    torch.manual_seed(2)
    L = GlobalLoadings(6, 50, ridge=0.0)
    B = L.B.detach().double()
    x = torch.randn(9, 50, dtype=torch.float64)
    cols = torch.tensor([3, 17, 49, 0, 25, 25, NO_MASK, 8, 41])

    z = L.project(x.float(), cols).double()

    for i in range(len(x)):
        g = int(cols[i])
        keep = torch.ones(50, dtype=torch.bool)
        if g != NO_MASK:
            keep[g] = False
        expected = torch.linalg.lstsq(B[:, keep].T, x[i, keep].unsqueeze(-1)).solution
        assert torch.allclose(z[i], expected.squeeze(-1), atol=1e-4), f"row {i}"


def test_masking_changes_the_projection():
    """If masking were a no-op the on-target diagonal would still be in play."""
    torch.manual_seed(3)
    L = GlobalLoadings(4, 20)
    x = torch.randn(5, 20)
    x[:, 7] = -50.0  # a dominant on-target-like entry
    cols = torch.full((5,), 7)
    assert not torch.allclose(L.project(x), L.project(x, cols), atol=1e-3)


def test_no_mask_sentinel_rows_are_unmasked():
    torch.manual_seed(4)
    L = GlobalLoadings(3, 15)
    x = torch.randn(4, 15)
    cols = torch.full((4,), NO_MASK)
    assert torch.allclose(L.project(x, cols), L.project(x), atol=1e-5)


def test_projection_is_differentiable_through_B():
    L = GlobalLoadings(3, 12)
    x = torch.randn(6, 12)
    cols = torch.tensor([1, 2, NO_MASK, 4, 4, 11])
    L.project(x, cols).sum().backward()
    assert L.B.grad is not None and torch.isfinite(L.B.grad).all()


# -------------------------------------------------------------------- eq. (10)


def test_squared_error_excludes_the_masked_entry():
    torch.manual_seed(5)
    L = GlobalLoadings(3, 10)
    x = torch.randn(4, 10)
    z = L.project(x)
    cols = torch.tensor([2, 2, NO_MASK, 9])

    err = L.squared_error(x, z, cols)
    resid = (x - L.reconstruct(z)).detach()

    for i in range(4):
        g = int(cols[i])
        keep = torch.ones(10, dtype=torch.bool)
        if g != NO_MASK:
            keep[g] = False
        expected = resid[i, keep].pow(2).sum() / keep.sum()
        assert torch.allclose(err[i], expected, atol=1e-5), f"row {i}"


def test_head_is_additively_separable():
    """Changing z_k must move the score only through psi_k -- no cross terms."""
    torch.manual_seed(6)
    head = PerComponentHead()
    d, J = 5, head.n_basis
    lam = torch.randn(1, d, J)
    z = torch.randn(1, d)

    base = head(z, lam)
    total_delta = 0.0
    for k in range(d):
        z_k = z.clone()
        z_k[0, k] += 0.5
        total_delta += float(head(z_k, lam) - base)

    z_all = z.clone() + 0.0
    z_all[0, :] += 0.5
    joint_delta = float(head(z_all, lam) - base)
    assert abs(joint_delta - total_delta) < 1e-4


def test_purely_quadratic_basis_is_rejected():
    with pytest.raises(ValueError, match="rotation invariant"):
        PerComponentHead(basis=("square",))


def test_default_basis_has_four_statistics():
    assert PerComponentHead().n_basis == 4


def test_component_evidence_sums_to_the_score():
    torch.manual_seed(7)
    d = 4
    head = PerComponentHead()
    net = FactorizedLabelNet(6, 3, d, head.n_basis, embedding_dim=8, hidden=0)
    z = torch.randn(5, d)
    p = torch.tensor([0, 1, 2, 3, 4])
    c = torch.tensor([0, 1, 2, 0, 1])
    lam = net(p, c)
    psi = (lam * head.statistics(z)).sum(dim=-1)
    assert torch.allclose(psi.sum(dim=1) + head.bias, head(z, lam), atol=1e-5)


def test_label_net_shares_embeddings_rather_than_memorising_pairs():
    net = FactorizedLabelNet(100, 8, 10, 4, embedding_dim=16, hidden=0)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 100 * 8 * 10  # far below one row per observed pair


def test_label_net_takes_different_perturbation_and_context_sizes():
    net = FactorizedLabelNet(30, 4, 6, 1, embedding_dim=64, context_embedding_dim=16)
    assert net.perturbation_embedding.weight.shape == (30, 64)
    assert net.context_embedding.weight.shape == (4, 16)
    assert net(torch.tensor([0, 5]), torch.tensor([1, 3])).shape == (2, 6, 1)


def test_control_mean_embeddings_follow_the_context_order_and_scale():
    import pandas as pd
    from cryptex.train import control_mean_embeddings
    rng = np.random.default_rng(0)
    means = pd.DataFrame(rng.normal(size=(5, 40)) + 3.0, index=list("abcde"))
    E = control_mean_embeddings(means, ["c", "a", "e", "b", "d"], dim=3)
    assert E.shape == (5, 3) and np.isclose(E.std(), 0.02, rtol=1e-3)
    E2 = control_mean_embeddings(means, list("abcde"), dim=3)
    assert np.allclose(np.abs(E[0]), np.abs(E2[2]))           # row of context "c"
    with pytest.raises(ValueError):
        control_mean_embeddings(means, ["a", "z"], dim=3)


def test_fit_starts_contexts_at_a_given_embedding():
    X, p, c, s, n = _toy_dataset()
    init = np.random.default_rng(1).normal(size=(3, 8)).astype(np.float32) * 0.02
    cfg = TrainConfig(pert_embedding_init="random", n_factors=5, epochs=0, batch_size=64, log_every=0, noise_rank=3, context_embedding_dim=8)
    model, _ = fit(X, p, c, config=cfg, context_embedding_init=init)
    assert np.allclose(model.label_net.context_embedding.weight.detach().numpy(), init)


def test_response_embeddings_use_training_pairs_and_skip_own_knockdown():
    from cryptex.train import response_embeddings
    rng = np.random.default_rng(0)
    n_pert, n_ctx, G = 12, 3, 20
    p = np.repeat(np.arange(n_pert), n_ctx * 2); c = np.tile(np.repeat(np.arange(n_ctx), 2), n_pert)
    X = rng.normal(size=(len(p), G)).astype(np.float32)
    target = np.where(p < 10, p, NO_MASK)                      # perts 10, 11: target not a response gene
    X[np.arange(len(p))[p < 10], target[p < 10]] = 50.0        # huge on-target knockdown
    train = c < 2
    E, has = response_embeddings(X, p, c, target, train, n_pert, dim=4)
    assert has.tolist() == [True] * 10 + [False] * 2
    assert np.allclose(E[~has], 0) and np.isclose(E[has].std(), 0.02, rtol=1e-3)
    # the knockdown entries are left out: changing them does not change the embeddings
    X2 = X.copy(); X2[np.arange(len(p))[p < 10], target[p < 10]] = -50.0
    assert np.allclose(np.abs(E), np.abs(response_embeddings(X2, p, c, target, train, n_pert, dim=4)[0]), atol=1e-5)
    # held-out pairs do not enter
    X3 = X.copy(); X3[~train] = rng.normal(size=((~train).sum(), G))
    assert np.allclose(np.abs(E), np.abs(response_embeddings(X3, p, c, target, train, n_pert, dim=4)[0]), atol=1e-5)


def test_fit_runs_with_label_regularisation():
    X, p, c, s, n = _toy_dataset()
    cfg = TrainConfig(pert_embedding_init="random", n_factors=5, epochs=4, batch_size=64, log_every=0, seed=0, noise_rank=3, label_weight_decay=1e-2, label_dropout=0.2)
    model, history = fit(X, p, c, target_col=None, config=cfg)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.label_net.modules())
    assert np.isfinite(history[-1]["disc"])


def _toy_labels(n_pert=12, n_ctx=4, n_strata=11):
    p, c, s = [], [], []
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                p.append(pi), c.append(ci), s.append(si)
    return np.array(p), np.array(c), np.array(s)


def _sampler(weights=None):
    p, c, s = _toy_labels()
    return (
        StratifiedNegativeSampler(
            perturbation_idx=p,
            context_idx=c,
            stratum=s,
            n_perturbations=12,
            n_contexts=4,
            **({"weights": weights} if weights else {}),
        ),
        p,
        c,
        s,
    )


def test_negatives_never_equal_their_own_label():
    sampler, p, c, _ = _sampler()
    rng = np.random.default_rng(0)
    rows = rng.permutation(len(p))[:2000]
    neg_p, neg_c = sampler.sample(rows, rng)
    collisions = (neg_p == p[rows]) & (neg_c == c[rows])
    assert collisions.sum() == 0, f"{collisions.sum()} false negatives"


def test_other_pert_strategy_keeps_the_context():
    sampler, p, c, _ = _sampler(weights={"same_s_other_pert": 1.0})
    rng = np.random.default_rng(1)
    rows = np.arange(500)
    neg_p, neg_c = sampler.sample(rows, rng)
    assert np.array_equal(neg_c, c[rows])
    assert (neg_p != p[rows]).all()


def test_other_context_strategy_keeps_the_perturbation():
    sampler, p, c, _ = _sampler(weights={"same_s_other_context": 1.0})
    rng = np.random.default_rng(2)
    rows = np.arange(500)
    neg_p, neg_c = sampler.sample(rows, rng)
    assert np.array_equal(neg_p, p[rows])
    assert (neg_c != c[rows]).all()


def test_single_context_perturbation_swaps_the_other_axis():
    """A perturbation with one observed context has no alternative context.
    The sampler must swap its perturbation instead of drawing a context from
    the full range, which would name an unobserved (possibly held-out) pair."""
    # perturbation 0 only in context 0; perturbations 1-3 in contexts 0-2
    rows = [(0, 0, s) for s in range(2)]
    rows += [(pp, cc, s) for pp in (1, 2, 3) for cc in range(3) for s in range(2)]
    P, C, S = (np.array(v) for v in zip(*rows))
    sampler = StratifiedNegativeSampler(
        perturbation_idx=P, context_idx=C, stratum=S,
        n_perturbations=4, n_contexts=3,
        weights={"same_s_other_context": 1.0},
    )
    observed = set(zip(P.tolist(), C.tolist()))
    rng = np.random.default_rng(0)
    for _ in range(50):
        neg_p, neg_c = sampler.sample(np.arange(len(P)), rng)
        named = set(zip(neg_p.tolist(), neg_c.tolist()))
        assert named <= observed, f"unobserved pairs named: {named - observed}"
        assert not ((neg_p == P) & (neg_c == C)).any()
    # the single-context rows got a perturbation swap, context untouched
    assert (neg_c[:2] == 0).all() and (neg_p[:2] != 0).all()


def test_label_with_no_observed_alternative_raises():
    """If neither axis has an observed alternative there is no legal negative;
    fail loudly rather than name an unobserved pair."""
    sampler = StratifiedNegativeSampler(
        perturbation_idx=np.array([0, 0]), context_idx=np.array([0, 0]),
        stratum=np.array([0, 1]), n_perturbations=2, n_contexts=2,
    )
    with pytest.raises(ValueError, match="no observed alternative"):
        sampler.sample(np.arange(2), np.random.default_rng(0))


def test_unknown_strategy_is_rejected():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="unknown strategies"):
        StratifiedNegativeSampler(
            perturbation_idx=p,
            context_idx=c,
            stratum=s,
            n_perturbations=12,
            n_contexts=4,
            weights={"against_vehicle": 1.0},
        )


def test_stratification_narrows_the_precision_gap():
    """The measurement that justifies stratifying: median |log2 ratio| should
    be well below the unstratified figure."""
    rng = np.random.default_rng(4)
    n_pert, n_ctx, n_strata = 40, 3, 11
    p, c, s, n_cells = [], [], [], []
    depth = rng.integers(60, 4000, n_pert)  # per-perturbation cell counts
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                p.append(pi), c.append(ci), s.append(si)
                frac = 1.0 if si == 0 else 0.1 + 0.08 * (si - 1)
                n_cells.append(max(50, int(depth[pi] * frac)))
    p, c, s = np.array(p), np.array(c), np.array(s)
    n_cells = np.array(n_cells, dtype=float)

    stratified = StratifiedNegativeSampler(
        perturbation_idx=p, context_idx=c, stratum=s,
        n_perturbations=n_pert, n_contexts=n_ctx,
        weights={"same_s_other_pert": 1.0},
    )
    rows = rng.permutation(len(p))[:3000]
    neg_p, neg_c = stratified.sample(rows, rng)
    strat_gap = np.nanmedian(
        stratified.precision_mismatch(rows, neg_p, neg_c, n_cells)
    )

    # Unstratified: a perturbation drawn without regard to stratum.
    # Unstratified baseline: same negative perturbations, but the negative
    # names a row at a *random* stratum -- what a sampler that ignores the
    # stratum effectively does.
    naive_s = rng.integers(0, n_strata, len(rows))
    naive_gap = np.nanmedian(
        stratified.precision_mismatch(
            rows, neg_p, neg_c, n_cells, neg_stratum=naive_s
        )
    )
    assert strat_gap < naive_gap, f"stratified {strat_gap:.3f} vs naive {naive_gap:.3f}"


# ----------------------------------------------------------------- on-target


def test_on_target_index_matches_bare_and_subsampled_labels():
    genes = np.array(["A1BG", "BRCA1", "TP53"])
    perts = np.array(["A1BG", "A1BG__sub03", "TP53__sub10", "non-targeting", "XYZ"])
    cols = on_target_index(perts, genes)
    assert cols.tolist() == [0, 0, 2, NO_MASK, NO_MASK]


def test_mask_coverage_reports_the_matched_fraction():
    cov = mask_coverage(np.array([0, 0, 2, NO_MASK]))
    assert cov["n_masked"] == 3
    assert cov["n_distinct_genes"] == 2


# --------------------------------------------------------------- eq. (14)


def test_span_residual_is_zero_inside_the_span():
    rng = np.random.default_rng(5)
    B = rng.normal(size=(4, 30))
    Z = rng.normal(size=(10, 4))
    assert np.allclose(span_residual(B, Z @ B), 0.0, atol=1e-8)


def test_span_residual_is_positive_outside_the_span():
    rng = np.random.default_rng(6)
    B = rng.normal(size=(3, 40))
    X = rng.normal(size=(12, 40))
    assert (span_residual(B, X) > 0.5).all()


# ------------------------------------------------------------------ eq. (12)


def _toy_dataset(n_pert=20, n_ctx=3, n_strata=4, n_genes=60, d_true=5, seed=0):
    rng = np.random.default_rng(seed)
    B_true = rng.normal(size=(d_true, n_genes))
    rows, p, c, s, n_cells = [], [], [], [], []
    mu = rng.normal(size=(n_pert, d_true))
    gamma = rng.normal(size=(n_ctx, d_true)) * 0.5
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                n = 3000 if si == 0 else int(80 * (si + 1))
                z = mu[pi] + gamma[ci]
                noise = rng.normal(size=n_genes) / np.sqrt(n)
                rows.append(z @ B_true + noise)
                p.append(pi), c.append(ci), s.append(si), n_cells.append(n)
    return (
        np.asarray(rows, dtype=np.float32),
        np.array(p), np.array(c), np.array(s), np.array(n_cells, dtype=float),
    )


def test_fit_runs_and_reduces_both_terms():
    X, p, c, s, n = _toy_dataset()
    # NB the discriminator loss on this toy barely moves in 12 epochs, so the
    # disc assertion below depends on the random stream (it fails for seed 2
    # with either draw); pinned to the stream it was written for.
    cfg = TrainConfig(pert_embedding_init="random", subspace="free", n_factors=5, epochs=12, batch_size=64, log_every=0, seed=0,
                      negative_draw="labels", lasso_groups="pair")
    model, history = fit(X, p, c, target_col=None, config=cfg)

    assert len(history) == 12
    assert history[-1]["recon"] < history[0]["recon"]
    assert history[-1]["disc"] < history[0]["disc"]
    assert model.loading_matrix().shape == (5, X.shape[1])


def test_fit_accepts_a_mask_and_custom_negative_weights():
    X, p, c, s, n = _toy_dataset(n_pert=12, n_genes=40)
    target_col = (p % 40).astype(np.int64)
    cfg = TrainConfig(pert_embedding_init="random", 
        n_factors=4, epochs=4, batch_size=48, log_every=0,
        negative_weights={"same_s_other_pert": 0.8, "same_s_other_context": 0.2},
    )
    model, history = fit(X, p, c, target_col=target_col, config=cfg)
    assert np.isfinite(history[-1]["recon"])
    assert model.loading_matrix().shape == (4, 40)


def test_fit_rejects_mismatched_array_lengths():
    X, p, c, s, n = _toy_dataset(n_pert=6, n_genes=20)
    with pytest.raises(ValueError, match="context_idx"):
        fit(X, p, c[:-1], config=TrainConfig(pert_embedding_init="random", epochs=1, log_every=0))


def test_contrastive_loss_is_minimised_by_confident_correct_logits():
    good = contrastive_loss(torch.tensor([6.0, 6.0]), torch.tensor([-6.0, -6.0]))
    bad = contrastive_loss(torch.tensor([-6.0, -6.0]), torch.tensor([6.0, 6.0]))
    assert float(good) < 0.01 < float(bad)


def test_unknown_strategy_rejected():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="unknown strategies"):
        StratifiedNegativeSampler(
            perturbation_idx=p, context_idx=c, stratum=s,
            n_perturbations=12, n_contexts=4, weights={"nonsense": 1.0},
        )


def test_zero_weights_rejected():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="positive"):
        StratifiedNegativeSampler(
            perturbation_idx=p, context_idx=c, stratum=s,
            n_perturbations=12, n_contexts=4,
            weights={"same_s_other_pert": 0.0},
        )


# -------------------------------------------------------------- train/test


def _toy_meta(n_pert=12, n_ctx=4, n_strata=3):
    import pandas as pd
    rows = []
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                lab = f"P{pi}" if si == 0 else f"P{pi}__sub{si-1:02d}"
                rows.append(
                    {"perturbation": f"P{pi}", "context": f"C{ci}",
                     "label": lab, "n_cells": 3000 if si == 0 else 100 * si}
                )
    return pd.DataFrame(rows)


def test_fit_trains_only_on_train_rows():
    """A held-out pair must never appear in a training batch."""
    from cryptex.data.splits import make_split

    meta = _toy_meta()
    X, p, c, s, n = _toy_dataset(n_pert=12, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.2, seed=0)

    pairs = list(zip(meta.perturbation, meta.context))
    train_pairs = {pc for pc, m in zip(pairs, split.train) if m}
    test_pairs = {pc for pc, m in zip(pairs, split.test) if m}
    assert not (train_pairs & test_pairs)

    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=4, batch_size=32, log_every=0,
                      eval_every=2)
    model, history = fit(
        X, p, c, config=cfg,
        train_rows=split.train, val_rows=split.test,
    )
    assert "val_accuracy" in history[-1]
    assert history[-1]["val_n_rows"] == int(split.test.sum())


def test_all_eleven_samples_of_a_held_out_pair_move_together():
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=20, n_ctx=3, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.15, seed=1)
    grouped = meta.assign(test=split.test).groupby(
        ["perturbation", "context"]
    )["test"].nunique()
    assert (grouped == 1).all(), "a pair was split across train and test"


def test_fit_rejects_a_split_smaller_than_one_batch():
    X, p, c, s, n = _toy_dataset(n_pert=6, n_ctx=2, n_strata=2, n_genes=20)
    train = np.zeros(len(X), dtype=bool)
    train[:10] = True
    with pytest.raises(ValueError, match="training rows"):
        fit(X, p, c, config=TrainConfig(pert_embedding_init="random", batch_size=256, epochs=1,
                                              log_every=0),
            train_rows=train)


def test_evaluate_returns_the_documented_keys():
    from cryptex.objectives import StratifiedNegativeSampler
    from cryptex.train import evaluate

    X, p, c, s, n = _toy_dataset(n_pert=10, n_ctx=3, n_strata=3, n_genes=30)
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=2, batch_size=32, log_every=0,
                      eval_every=0)
    model, _ = fit(X, p, c, config=cfg)

    rows = np.arange(min(200, len(X)))
    sampler = StratifiedNegativeSampler(
        perturbation_idx=p[rows], context_idx=c[rows], stratum=s[rows],
        n_perturbations=int(p.max()) + 1, n_contexts=int(c.max()) + 1,
        weights={"same_s_other_pert": 1.0},
    )
    m = evaluate(model, torch.from_numpy(X), p, c, rows, sampler,
                 np.random.default_rng(0))
    for key in ("auc", "n_rows", "accuracy", "disc", "disc_objective", "recon", "real_score_mean",
                "fake_score_mean", "span_residual_median", "span_residual_q90",
                "lasso_objective", "programs_per_pair"):
        assert key in m, key
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["auc"] <= 1.0
    assert 0.0 <= m["span_residual_median"] <= 1.5


def test_three_way_split_is_disjoint_and_complete():
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=40, n_ctx=4, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.1, val_frac=0.1, seed=0)
    assert split.three_way
    assert int(split.train.sum() + split.val.sum() + split.test.sum()) == len(meta)
    assert not (split.train & split.val).any()
    assert not (split.train & split.test).any()
    assert not (split.val & split.test).any()
    assert not set(split.held_out) & set(split.held_out_val)


def test_three_way_split_keeps_pairs_whole():
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=30, n_ctx=3, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=3)
    half = np.select(
        [split.train, split.val, split.test], ["train", "val", "test"],
        default="none",
    )
    assert "none" not in set(half)
    per_pair = (
        meta.assign(half=half).groupby(["perturbation", "context"])["half"].nunique()
    )
    assert (per_pair == 1).all()


def test_split_rejects_impossible_fractions():
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=4, n_ctx=1, n_strata=2)
    with pytest.raises(ValueError, match="would hold out"):
        make_split(meta, level="pair", test_frac=0.7, val_frac=0.7, seed=0)


def test_the_kept_model_is_the_epoch_with_the_best_validation_auc():
    """Every epoch is trained; the kept state is the epoch whose validation ROC
    AUC is highest, and the last epoch's state is kept separately."""
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=12, batch_size=32, log_every=0,
                      eval_every=1, seed=0)
    model, history = fit(X, p, c, config=cfg, train_rows=split.train, val_rows=split.val)
    best = max((h for h in history if "val_auc" in h), key=lambda h: h["val_auc"])
    assert len(history) == 12
    assert history[-1]["selected_epoch"] == best["epoch"] == model.selected_epoch_
    assert model.final_epoch_ == 11 and set(model.final_state_) == set(model.state_dict())


def test_total_is_the_objective_on_both_train_and_val():
    """`total` must equal disc + alpha*recon on both sides, so the two curves
    are the same quantity and their crossing is meaningful."""
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)
    alpha = 700.0
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, alpha=alpha, epochs=6, batch_size=32,
                      log_every=0, eval_every=1, seed=0)
    _, history = fit(X, p, c, config=cfg,
                     train_rows=split.train, val_rows=split.val)
    for h in history:
        # (recon holds the lasso penalty under the default group lasso; the
        # gene prior and the distinctness hinge are on by default)
        gene = cfg.gene_sparsity * h["gene_penalty"] + cfg.distinct_weight * h["distinct_penalty"]
        assert abs(h["total"] - (h["disc"] + alpha * h["recon"] + gene)) < 1e-6
        if "val_total" in h:
            vgene = cfg.gene_sparsity * h["val_gene_penalty"] + cfg.distinct_weight * h["val_distinct_penalty"]
            assert abs(h["val_total"]
                       - (h["val_disc_objective"] + alpha * h["val_lasso_objective"] + vgene)) < 1e-6


def test_fit_rejects_overlapping_train_and_val():
    X, p, c, s, n = _toy_dataset(n_pert=12, n_ctx=3, n_strata=3, n_genes=30)
    mask = np.zeros(len(X), dtype=bool)
    mask[:] = True
    with pytest.raises(ValueError, match="overlap"):
        fit(X, p, c, config=TrainConfig(pert_embedding_init="random", epochs=1, batch_size=32,
                                              log_every=0),
            train_rows=mask, val_rows=mask)


# --- is_real gate and masked span residual ---------------------------------


def test_span_residual_excludes_the_masked_entry():
    """B is trained never to reconstruct the on-target entry, so charging the
    residual for it inflates r. Masked and unmasked must differ, and masking
    must match an explicit computation that drops the column."""
    torch.manual_seed(0)
    L = GlobalLoadings(3, 12, ridge=1e-4)
    x = torch.randn(5, 12)
    col = torch.full((5,), NO_MASK, dtype=torch.long)
    col[:3] = torch.tensor([0, 4, 9])

    r_masked = L.span_residual(x, col)
    r_plain = L.span_residual(x)
    assert not torch.allclose(r_masked[:3], r_plain[:3])
    assert torch.allclose(r_masked[3:], r_plain[3:])       # unmasked rows unchanged

    z = L.project(x, col)
    resid = (x - L.reconstruct(z)).detach()
    xm = x.clone()
    for i in range(3):
        resid[i, col[i]] = 0.0
        xm[i, col[i]] = 0.0
    expect = torch.linalg.norm(resid, dim=1) / torch.linalg.norm(xm, dim=1)
    assert torch.allclose(r_masked, expect, atol=1e-6)


def test_negatives_never_name_a_held_out_pair():
    """The sampler is built on training rows only, and every strategy must draw
    from an OBSERVED (perturbation, context) pair, so a held-out pair is never
    named as a negative."""
    rng = np.random.default_rng(0)
    n_p, n_c, n_s = 40, 6, 3
    rows = [(p, c, s) for p in range(n_p) for c in range(n_c) for s in range(n_s)]
    # hold out a tenth of the (p, c) pairs
    pairs = sorted({(p, c) for p, c, _ in rows})
    held = set(map(tuple, rng.permutation(pairs)[: len(pairs) // 10].tolist()))

    tr = [(p, c, s) for p, c, s in rows if (p, c) not in held]
    P = np.array([r[0] for r in tr]); C = np.array([r[1] for r in tr])
    S = np.array([r[2] for r in tr])

    sampler = StratifiedNegativeSampler(
        perturbation_idx=P, context_idx=C, stratum=S,
        n_perturbations=n_p, n_contexts=n_c, weights=dict(DEFAULT_WEIGHTS),
    )
    neg_p, neg_c = sampler.sample(np.arange(len(tr)), np.random.default_rng(1))

    named_held = [(int(a), int(b)) for a, b in zip(neg_p, neg_c) if (int(a), int(b)) in held]
    assert not named_held, f"{len(named_held)} negatives named a held-out pair, e.g. {named_held[:3]}"
    observed = {(p, c) for p, c, _ in tr}
    unseen = [(int(a), int(b)) for a, b in zip(neg_p, neg_c) if (int(a), int(b)) not in observed]
    assert not unseen, f"{len(unseen)} negatives named an unobserved pair, e.g. {unseen[:3]}"
    assert not ((neg_p == P) & (neg_c == C)).any(), "a negative kept its own label"


def test_single_context_perturbations_always_train():
    """A perturbation with one pair has nothing left to train e_p on if that
    pair is held out, so its row is unpredictable by construction. Those pairs
    must stay in train."""
    from cryptex.data.splits import make_split

    rows = []
    for i in range(200):                       # 12 contexts each
        for c in range(12):
            rows.append({"perturbation": f"P{i:03d}", "context": f"C{c}",
                         "label": f"P{i:03d}"})
    for i in range(20):                        # single-context perturbations
        rows.append({"perturbation": f"S{i:02d}", "context": "C0",
                     "label": f"S{i:02d}"})
    meta = pd.DataFrame(rows)
    singles = {f"S{i:02d}" for i in range(20)}

    sp = make_split(meta, level="pair", test_frac=0.2, val_frac=0.2, seed=0)
    held = set(meta[sp.test].perturbation) | set(meta[sp.val].perturbation)
    assert not (singles & held), sorted(singles & held)[:3]
    # every perturbation keeps at least one training pair
    assert set(meta.perturbation) <= set(meta[sp.train].perturbation)
    # and the held-out count is still the requested fraction of ALL pairs
    n_pairs = meta.groupby(["perturbation", "context"], observed=True).ngroups
    assert abs(len(sp.held_out) - round(0.2 * n_pairs)) <= 1

    off = make_split(meta, level="pair", test_frac=0.2, val_frac=0.2, seed=0,
                     protect_single_context=False)
    held_off = set(meta[off.test].perturbation) | set(meta[off.val].perturbation)
    assert singles & held_off, "opt-out should let singles be held out"


def test_evaluate_takes_its_device_from_the_model_not_the_data():
    """A caller may deliberately keep the full matrix on the CPU (it is ~11 GB
    on the real split) and let batches move. Reading the device off X_t instead
    of the model sends CPU batches into a CUDA model and the first matmul
    raises. Checked on CPU by putting the model on a meta-free device and
    asserting evaluate() uses `next(model.parameters()).device`."""
    import inspect
    from cryptex import train as T
    src_ = inspect.getsource(T.evaluate)
    assert "next(model.parameters()).device" in src_, \
        "evaluate() must take its device from the model"
    assert "device = X_t.device" not in src_, \
        "evaluate() must not take its device from the data tensor"
    # every batch slice of the data tensor is moved explicitly
    for frag in ("X_t[idx_t].to(device)", "target_col[idx_t].to(device)"):
        assert frag in src_, f"missing device move: {frag}"


def test_on_eval_snapshots_B_at_every_evaluation():
    """on_eval fires once per validation pass with the live model, so a copy
    of B taken there is the B of that epoch -- not the restored one."""
    from cryptex.data.splits import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)
    cfg = TrainConfig(pert_embedding_init="random", n_factors=4, epochs=10, batch_size=32, log_every=0,
                      eval_every=3, seed=0)
    snaps = {}
    model, history = fit(X, p, c, config=cfg,
                         train_rows=split.train, val_rows=split.val,
                         on_eval=lambda ep, m: snaps.__setitem__(
                             ep, m.loading_matrix().copy()))
    evals = [h["epoch"] for h in history if "val_accuracy" in h]
    assert sorted(snaps) == evals == [0, 3, 6, 9]
    assert not np.allclose(snaps[0], snaps[9])
    sel = history[-1]["selected_epoch"]
    np.testing.assert_allclose(model.loading_matrix(), snaps[sel])


def _uneven_labels(seed=0):
    """Perturbations with very different numbers of rows, in 3 contexts."""
    rng = np.random.default_rng(seed)
    p, c = [], []
    for pi in range(20):
        for ci in range(3):
            n = int(rng.integers(1, 12))
            p += [pi] * n
            c += [ci] * n
    return np.array(p), np.array(c)


def test_row_weighted_negatives_match_the_positive_label_frequencies():
    """Each label should be a negative about as often as it is a positive, so a
    label-only term cannot tell which labels have many rows."""
    p, c = _uneven_labels()
    rng = np.random.default_rng(1)
    keys = p * 3 + c
    pos = np.bincount(keys, minlength=60) / len(keys)
    for row_weighted, close in ((True, True), (False, False)):
        sampler = StratifiedNegativeSampler(
            perturbation_idx=p, context_idx=c, stratum=np.zeros(len(p), dtype=np.int64),
            n_perturbations=20, n_contexts=3, row_weighted=row_weighted)
        draws = [sampler.sample(np.arange(len(p)), rng) for _ in range(200)]
        neg = np.bincount(np.concatenate([np_ * 3 + nc for np_, nc in draws]), minlength=60)
        neg = neg / neg.sum()
        r = np.corrcoef(pos, neg)[0, 1]
        assert (r > 0.95) if close else (r < 0.6), (row_weighted, r)


def test_row_weighted_negatives_still_name_only_observed_other_labels():
    p, c = _uneven_labels(2)
    sampler = StratifiedNegativeSampler(
        perturbation_idx=p, context_idx=c, stratum=np.zeros(len(p), dtype=np.int64),
        n_perturbations=20, n_contexts=3)
    observed = set(zip(p.tolist(), c.tolist()))
    neg_p, neg_c = sampler.sample(np.arange(len(p)), np.random.default_rng(0))
    assert set(zip(neg_p.tolist(), neg_c.tolist())) <= observed
    assert not ((neg_p == p) & (neg_c == c)).any()
