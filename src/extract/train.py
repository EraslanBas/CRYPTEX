"""EXTRACT: the model and its training loop.

    L = L_disc + alpha * L_lasso + gamma * L_genes + mu * L_distinct

``L_disc`` is the label-swap discriminator on the per-component head,
``L_lasso`` the lasso objective (reconstruction error plus the penalty; a
lasso per row by default), ``L_genes`` the L1/L2 sparseness of each program's loadings and
``L_distinct`` the hinge on near-collinear programs. See the documentation.
"""

from __future__ import annotations

from typing import Callable

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data.ontarget import on_target_index
from .models import (initial_axes, FactorizedLabelNet, FixedBasisLoadings, GlobalLoadings,
                     PerComponentHead)
from .models.heads import DEFAULT_BASIS
from .models.loadings import NO_MASK
from .objectives import (
    DEFAULT_WEIGHTS,
    StratifiedNegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
    weighted_squared_error,
)


@dataclass
class TrainConfig:
    #: Number of programs d. Dataset-specific; tune it.
    n_factors: int = 20

    #: Statistics of the per-component head. Must retain a non-quadratic one:
    #: a purely quadratic head is rotation invariant and identifies nothing.
    basis: tuple[str, ...] = DEFAULT_BASIS
    #: Sizes of the perturbation and context embeddings and of the label
    #: network's hidden layer.
    embedding_dim: int = 64
    context_embedding_dim: int = 8
    label_hidden: int = 128
    #: Conditioning of the least-squares projections.
    ridge: float = 1e-4

    #: Weight of the group-lasso objective (reconstruction error plus group
    #: penalty, per row and gene) against the discriminator loss.
    alpha: float = 700.0

    batch_size: int = 1024
    epochs: int = 1000
    lr: float = 1e-3
    grad_clip: float | None = 5.0
    seed: int = 0
    device: str = "cpu"

    #: How the span of B is set. ``"fixed"`` (the model): B = A V with V the
    #: subspace computed once from the training rows and only the d x d matrix
    #: ``A`` learned. ``"free"``: B learned directly.
    subspace: str = "fixed"

    #: How V is computed. ``"pca"``: PCA of the pair means (every row of a pair
    #: averaged; uncentred). ``"rca"``: reliable components analysis (the
    #: directions whose values agree most between different rows of the same
    #: pair, relative to their total variance; Dmochowski et al.), R_W truncated
    #: to its top ``rca_rank`` eigen-directions. ``"snr"``: between-pair signal
    #: over the within-pair noise model. See extract.data.noise.
    subspace_method: str = "pca"
    rca_rank: int = 100
    #: Rank of the correlated part of the noise model (``"snr"``), and the
    #: number of within-pair residuals used to fit it.
    noise_rank: int = 50
    noise_max_rows: int = 150_000

    #: Free mode only. ``"random"``: B starts at random. ``"basis"``: B starts
    #: where the fixed model starts, ``A0 V``, and is then learned freely.
    free_init: str = "random"

    #: Starting axes A: ``"jd"``, the joint-diagonalisation axes of the
    #: per-context covariances of the training pair means inside V (see
    #: :func:`jd_axes`), or ``"random"``, a random rotation of V's axes drawn
    #: from ``seed``.
    axes_init: str = "jd"

    #: Activities z. ``"group_lasso"``: a sparse fit through B
    #: (models.group_lasso), by default a lasso per row (``lasso_groups``);
    #: the loadings are kept at unit length and the reconstruction term is the
    #: lasso objective (error + lasso_lambda * penalty), so it prefers axes on
    #: which responses use few programs. ``"ls"``: masked least squares onto
    #: B, for comparison.
    coding: str = "group_lasso"
    lasso_lambda: float = 0.0272
    lasso_steps: int = 30
    #: Which rows share a selection of programs. ``"row"`` (default): each row
    #: is its own group (a lasso per row), so a row's activities depend on that
    #: row alone, not on the other rows of its pair or on how many there are --
    #: every row is one data point for the discriminator. ``"pair"``: all rows
    #: of a (perturbation, context) pair share one selection (the group lasso;
    #: runs before 2026-10-28).
    lasso_groups: str = "row"

    #: Regularisation of the label network against memorising training pairs:
    #: decoupled (AdamW) weight decay on its parameters only, and dropout on
    #: its embeddings and hidden layer. 0 = off.
    label_weight_decay: float = 0.01
    label_dropout: float = 0.2

    #: Starting perturbation embeddings. ``"response"``: from how each
    #: perturbation's target gene responds, as a response gene, across the
    #: training pairs (:func:`response_embeddings`); perturbations whose target
    #: is not a response gene keep a random start. ``"random"``. Trained further
    #: either way.
    pert_embedding_init: str = "response"

    #: Gene-side prior: few genes per program. gamma * sum_k ||b_k||_1 /
    #: (||b_k||_2 sqrt(G)) (heavy-tailed loadings), on the programs alone (no
    #: data rows), acting on A in the fixed mode. 0 = off.
    gene_sparsity: float = 0.05
    #: Keep programs distinct: mu * sum_{k<l} max(0, |cos(b_k, b_l)| - tau)^2.
    #: Zero unless two programs are nearly collinear; shared genes are free.
    #: Stops the gene prior from collapsing several programs onto the same genes.
    distinct_weight: float = 100.0
    distinct_tau: float = 0.5

    #: Ablation only: skips the on-target mask, so knockdown efficiency is free
    #: to claim a factor.
    no_mask: bool = False

    #: Evaluate on the validation rows every N epochs (0 = never). The kept
    #: state is the epoch with the highest validation ROC AUC.
    eval_every: int = 3

    negative_weights: dict[str, float] | None = None
    #: How a swapped label is drawn. ``"rows"``: in proportion to its number
    #: of rows, so every label is a negative about as often as it is a
    #: positive and a label-only term cannot learn which labels have many rows
    #: (cell counts). ``"labels"``: uniformly over the observed alternatives
    #: (runs before 2026-10-25).
    negative_draw: str = "rows"
    log_every: int = 10


class Extract(nn.Module):
    """One global loading matrix, a per-component head, and a label network.

    There is no encoder and no label prior. ``B`` is the only factor->gene map;
    ``z`` is fitted to ``x`` through ``B`` by the per-pair group lasso
    (``coding="group_lasso"``; masked least squares with ``coding="ls"``).
    """

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        n_contexts: int,
        config: TrainConfig,
        subspace_basis: np.ndarray | None = None,
        axes0: np.ndarray | None = None,
    ):
        super().__init__()
        A0 = None if axes0 is None else torch.as_tensor(np.asarray(axes0), dtype=torch.float32)
        self.config = config
        if config.subspace not in ("free", "fixed"):
            raise ValueError(f"subspace must be 'fixed' or 'free', got {config.subspace!r}")
        if config.subspace == "fixed" and subspace_basis is None:
            raise ValueError("subspace='fixed' needs subspace_basis")
        if config.subspace == "free" and config.free_init == "random":
            subspace_basis = None
        if subspace_basis is not None and np.asarray(subspace_basis).shape != (config.n_factors, n_genes):
            raise ValueError(f"subspace_basis must be [{config.n_factors}, {n_genes}]")
        if config.subspace == "fixed":
            self.loadings = FixedBasisLoadings(subspace_basis, ridge=config.ridge,
                                               axes_init=config.axes_init, seed=config.seed, A0=A0)
        else:
            self.loadings = GlobalLoadings(config.n_factors, n_genes, ridge=config.ridge)
            if config.free_init == "basis":
                if subspace_basis is None:
                    raise ValueError("free_init='basis' needs subspace_basis")
                if A0 is None:
                    A0 = initial_axes(config.n_factors, config.axes_init, config.seed)
                with torch.no_grad():
                    self.loadings.B.copy_(A0 @ torch.as_tensor(np.asarray(subspace_basis), dtype=torch.float32))
            elif config.free_init != "random":
                raise ValueError(f"free_init must be 'random' or 'basis', got {config.free_init!r}")
        self.head = PerComponentHead(config.basis, normaliser=True)
        self.label_net = FactorizedLabelNet(
            n_perturbations=n_perturbations,
            n_contexts=n_contexts,
            n_factors=config.n_factors,
            n_basis=self.head.n_outputs,
            embedding_dim=config.embedding_dim,
            context_embedding_dim=config.context_embedding_dim,
            hidden=config.label_hidden,
            dropout=config.label_dropout,
        )

    # ---- forward pieces ---------------------------------------------------

    def project(
        self, x: torch.Tensor, target_col: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The activities ``z``. Masked when ``target_col`` is given. With
        ``coding="group_lasso"`` the lasso activities (``groups`` [n]: the
        rows that share a selection; one per row by default, needed only for
        ``lasso_groups="pair"``), used both by the
        discriminator and in the reconstruction objective; they and the
        selection are also kept in ``lasso_raw_`` / ``lasso_sel_``."""
        z = self.loadings.project(x, target_col)
        if self.config.coding == "ls":
            return z
        if groups is None:
            if self.config.lasso_groups != "row":
                raise ValueError("the group lasso (lasso_groups='pair') needs the row groups")
            groups = torch.arange(len(x), device=x.device)   # a lasso per row
        from .models.group_lasso import group_lasso_codes
        Z, sel = group_lasso_codes(x, self.loadings.B, groups, self.config.lasso_lambda, target_col,
                                   z0=z.detach(), steps=self.config.lasso_steps)
        self.lasso_raw_, self.lasso_sel_ = Z, sel
        return Z

    def score(
        self,
        z: torch.Tensor,
        perturbation_idx: torch.Tensor,
        context_idx: torch.Tensor,
    ) -> torch.Tensor:
        """``s(x, u) = b + sum_k psi_k(z_k, u)``, the true-label logit."""
        return self.head(z, self.label_net(perturbation_idx, context_idx))

    def component_evidence(
        self,
        z: torch.Tensor,
        perturbation_idx: torch.Tensor,
        context_idx: torch.Tensor,
    ) -> torch.Tensor:
        """``psi_k`` per factor -- [batch, n_factors]. Per-factor evidence, not
        a per-factor decision: these are summed before any decision is taken."""
        return self.head.evidence(z, self.label_net(perturbation_idx, context_idx))

    @property
    def B(self) -> torch.Tensor:
        return self.loadings.B

    @torch.no_grad()
    def loading_matrix(self) -> np.ndarray:
        """``B`` as [n_factors, n_genes]. The interpretable output -- see
        ``interpret.loadings`` for sign anchoring, which is required before any
        directional claim."""
        return self.loadings.B.detach().cpu().numpy()


def jd_axes(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    rows: np.ndarray, V: np.ndarray,
) -> np.ndarray:
    """Starting axes ``A0`` [d, d] for ``B = A0 V``: the joint-diagonalisation
    axes of the per-context covariances of the training pair means, inside the
    span of V.

    The pair means are written in an orthonormal basis Q of V's span, ``y``.
    For every context, the covariance of ``y`` over its pairs (centred within
    the context) is ``S_c``; with ``W = (mean_c S_c)^{-1/2}``, an orthogonal
    ``U`` makes every ``W S_c W`` as diagonal as possible (Jacobi joint
    diagonalisation). The programs are the columns of the mixing ``W^{-1} U``,
    i.e. ``B = U^T W^{-1} Q^T``, rows scaled to unit length. Uses the training
    rows only."""
    from baselines.joint_diagonalization import joint_diagonalize
    from .data.noise import pair_means
    rows = np.nonzero(rows)[0] if np.asarray(rows).dtype == bool else np.asarray(rows)
    n_c = int(context_idx.max()) + 1
    keys, means, _ = pair_means(X, perturbation_idx * n_c + context_idx, rows)
    ctx = keys % n_c
    V = np.asarray(V, dtype=np.float64)
    Q = np.linalg.qr(V.T)[0]                                     # [G, d]
    Y = means @ Q
    covs = []
    for c in np.unique(ctx):
        Yc = Y[ctx == c]
        if len(Yc) > V.shape[0]:
            Yc = Yc - Yc.mean(0)
            covs.append(Yc.T @ Yc / len(Yc))
    if len(covs) < 2:
        raise ValueError("jd_axes needs at least two contexts with more pairs than factors")
    ev, E = np.linalg.eigh(np.mean(covs, 0))
    W = E @ np.diag(ev ** -0.5) @ E.T
    U, _ = joint_diagonalize(np.stack([W @ S @ W for S in covs]))
    Bq = U.T @ np.linalg.inv(W)                                  # rows: programs in Q coordinates
    B = Bq @ Q.T
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return (B @ np.linalg.pinv(V)).astype(np.float32)          # A0 with A0 V = B


def response_embeddings(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    target_col: np.ndarray, rows: np.ndarray, n_perturbations: int, dim: int,
    seed: int = 0, scale: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturbation embeddings from how each target gene responds.

    Transpose the training data: one row per target gene, one column per
    training pair, holding that gene's response (the pair's mean over its
    measured rows) in that pair. Entries where the pair's own perturbation
    targets the gene are left out (set to 0) -- the knockdown would dominate.
    A truncated SVD of these rows gives ``dim`` numbers per target gene,
    rescaled so the entries have standard deviation ``scale`` (the size of the
    random start). Uses training pairs only, so validation and test pairs never
    shape an embedding. Returns ``(embeddings [n_perturbations, dim], has [n])``;
    ``has`` is False for perturbations whose target is not a response gene.
    """
    from sklearn.utils.extmath import randomized_svd
    from .data.noise import pair_means
    rows = np.nonzero(rows)[0] if np.asarray(rows).dtype == bool else np.asarray(rows)
    pair = perturbation_idx * (int(context_idx.max()) + 1) + context_idx
    keys, means, _ = pair_means(X, pair, rows)
    pair_pert = keys // (int(context_idx.max()) + 1)
    col_of = np.full(n_perturbations, NO_MASK, dtype=np.int64)
    ok = target_col != NO_MASK
    col_of[perturbation_idx[ok]] = target_col[ok]
    has = col_of != NO_MASK
    if not has.any():          # no target among the response genes: all keep the random start
        return np.zeros((n_perturbations, dim), dtype=np.float32), has
    T = means[:, col_of[has]].T.copy()                       # [targets, training pairs]
    T[pair_pert[None, :] == np.nonzero(has)[0][:, None]] = 0.0   # own knockdown entries
    U, S, _ = randomized_svd(T.astype(np.float32), min(dim, *T.shape), n_iter=7, random_state=seed)
    E = np.zeros((n_perturbations, dim), dtype=np.float32)
    Z = U * S
    E[has, :Z.shape[1]] = (Z / Z.std() * scale).astype(np.float32)
    return E, has


def control_mean_embeddings(means, context_levels, dim: int, scale: float = 0.02) -> np.ndarray:
    """Context embeddings [len(context_levels), dim] from the mean expression of
    each context's control cells: PCA of the contexts x genes matrix (centred
    across contexts, i.e. each gene minus its average over the contexts -- not
    relative to DMSO), scores on the top ``dim`` components, rescaled so the
    entries have standard deviation ``scale`` (the size of the random start).
    ``means`` is a DataFrame indexed by context name."""
    import pandas as pd
    M = pd.DataFrame(means).reindex([str(c) for c in context_levels])
    if M.isna().any().any():
        missing = [c for c in context_levels if c not in means.index]
        raise ValueError(f"no control means for contexts {missing}")
    Z = M.to_numpy(dtype=np.float64)
    Z = Z - Z.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    k = min(dim, len(S))
    E = np.zeros((len(Z), dim))
    E[:, :k] = U[:, :k] * S[:k]
    return (E / E[:, :k].std() * scale).astype(np.float32)


def fit(
    X: np.ndarray,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    target_col: np.ndarray | None = None,
    config: TrainConfig | None = None,
    train_rows: np.ndarray | None = None,
    val_rows: np.ndarray | None = None,
    on_eval: Callable[[int, Extract], None] | None = None,
    subspace_basis: np.ndarray | None = None,
    context_embedding_init: np.ndarray | None = None,
) -> tuple[Extract, list[dict]]:
    """Fit the model. Returns ``(model, history)``.

    Parameters
    ----------
    X
        [n_rows, n_genes] shrunken logFC, used **as it is**. Do not centre or
        scale the gene axis: ``ashr`` has already placed every entry on a
        common log-fold-change scale, and dividing by per-gene SD would remove
        exactly that and promote low-variance genes to equal footing.

        The model sees only ``X`` and the labels. All rows of a pair are
        exchangeable estimates of the same response: no cell counts, no
        subsample index, and no row is treated as the most precise.
    train_rows, val_rows
        Boolean masks over rows, split at the (perturbation, context) pair level
        (:func:`extract.data.splits.split_val_per_perturbation`, or
        :func:`extract.data.splits.make_split` for an in-memory table). All
        rows of a held-out pair move together. ``None`` trains on everything.

        **Validation, not test.** ``fit`` looks at ``val_rows`` repeatedly, to
        choose the epoch. The test rows must not be passed here.

        The negative sampler is built from the **training rows only**, so a
        held-out pair is never named even as a negative.
    on_eval
        Called as ``on_eval(epoch, model)`` after every validation pass, before
        the best state is restored. Used to snapshot ``B`` at each evaluation.
    subspace_basis
        A precomputed V [d, n_genes]; by default V is computed from the
        training rows (``config.subspace_method``).
    context_embedding_init
        Starting context embeddings [n_contexts, context_embedding_dim].
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    X = np.asarray(X, dtype=np.float32)
    perturbation_idx = np.asarray(perturbation_idx, dtype=np.int64)
    context_idx = np.asarray(context_idx, dtype=np.int64)
    n_rows, n_genes = X.shape

    for name, arr in (
        ("perturbation_idx", perturbation_idx),
        ("context_idx", context_idx),
    ):
        if len(arr) != n_rows:
            raise ValueError(f"{name} has {len(arr)} entries, X has {n_rows} rows")

    n_perturbations = int(perturbation_idx.max()) + 1
    n_contexts = int(context_idx.max()) + 1

    train_idx = (
        np.arange(n_rows)
        if train_rows is None
        else np.nonzero(np.asarray(train_rows, dtype=bool))[0]
    )
    val_idx = (
        None
        if val_rows is None
        else np.nonzero(np.asarray(val_rows, dtype=bool))[0]
    )
    if val_idx is not None and np.intersect1d(train_idx, val_idx).size:
        raise ValueError("train_rows and val_rows overlap")
    if len(train_idx) < config.batch_size:
        raise ValueError(
            f"only {len(train_idx)} training rows for batch_size "
            f"{config.batch_size}"
        )

    needs_basis = config.subspace == "fixed" or (
        config.subspace == "free" and config.free_init == "basis")
    if needs_basis and subspace_basis is None:
        # the subspace V, from the TRAINING rows only
        from .data.noise import (label_subspace_from_rows, pca_subspace_from_rows,
                                 rca_subspace_from_rows)
        if config.subspace_method not in ("rca", "snr", "pca"):
            raise ValueError(f"subspace_method must be 'rca', 'snr' or 'pca', got {config.subspace_method!r}")
        if config.subspace_method == "pca":
            subspace_basis = pca_subspace_from_rows(
                X, perturbation_idx, context_idx, train_idx, d=config.n_factors,
                seed=config.seed)
            what = "PCA of the pair means"
        elif config.subspace_method == "rca":
            subspace_basis = rca_subspace_from_rows(
                X, perturbation_idx, context_idx, train_idx, d=config.n_factors,
                rank=config.rca_rank, seed=config.seed)
            what = f"RCA, R_W rank {config.rca_rank}"
        else:
            subspace_basis, nm = label_subspace_from_rows(
                X, perturbation_idx, context_idx, train_idx,
                d=config.n_factors, rank=config.noise_rank,
                max_rows=config.noise_max_rows, seed=config.seed)
            what = f"signal/noise, noise rank {len(nm.s)}"
        if config.log_every:
            print(f"subspace: d={config.n_factors} ({what}), from the "
                  f"training rows", flush=True)

    if config.axes_init not in ("jd", "random"):
        raise ValueError(f"axes_init must be 'jd' or 'random', got {config.axes_init!r}")
    axes0 = None
    if config.axes_init == "jd" and subspace_basis is not None:
        # (free mode with a random B has no axes to start: nothing to do)
        axes0 = jd_axes(X, perturbation_idx, context_idx, train_idx, subspace_basis)
    model = Extract(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        n_contexts=n_contexts,
        config=config,
        subspace_basis=subspace_basis,
        axes0=axes0,
    ).to(device)
    if config.pert_embedding_init == "response":
        if target_col is None:
            raise ValueError("pert_embedding_init='response' needs target_col (the target gene's column)")
        emb, has = response_embeddings(
            X, perturbation_idx, context_idx, np.asarray(target_col), train_idx,
            n_perturbations, config.embedding_dim, seed=config.seed)
        with torch.no_grad():
            w = model.label_net.perturbation_embedding.weight
            w[torch.from_numpy(has)] = torch.from_numpy(emb[has]).to(w)
        if config.log_every:
            print(f"perturbation embeddings: {int(has.sum())} of {n_perturbations} from "
                  f"target-gene responses", flush=True)
    if context_embedding_init is not None:
        w = model.label_net.context_embedding.weight
        if tuple(np.shape(context_embedding_init)) != tuple(w.shape):
            raise ValueError(f"context_embedding_init must be {tuple(w.shape)}, "
                             f"got {np.shape(context_embedding_init)}")
        with torch.no_grad():
            w.copy_(torch.as_tensor(np.asarray(context_embedding_init), dtype=torch.float32))
    if config.pert_embedding_init not in ("random", "response"):
        raise ValueError(f"pert_embedding_init must be 'random' or 'response', "
                         f"got {config.pert_embedding_init!r}")
    model.subspace_basis_ = subspace_basis

    weights = dict(config.negative_weights or DEFAULT_WEIGHTS)
    if config.negative_draw not in ("rows", "labels"):
        raise ValueError(f"negative_draw must be 'rows' or 'labels', got {config.negative_draw!r}")

    def _sampler(rows: np.ndarray) -> StratifiedNegativeSampler:
        return StratifiedNegativeSampler(
            perturbation_idx=perturbation_idx[rows],
            context_idx=context_idx[rows],
            stratum=np.zeros(len(rows), dtype=np.int64),   # rows are exchangeable
            n_perturbations=n_perturbations,
            n_contexts=n_contexts,
            weights=weights,
            row_weighted=config.negative_draw == "rows",
        )

    # Built on the training rows only: a held-out pair is never named, even as
    # a negative. The sampler indexes the rows it was given, so its positions
    # are offsets into `train_idx` rather than into X.
    sampler = _sampler(train_idx)
    eval_sampler = _sampler(val_idx) if val_idx is not None else None

    w = np.ones(n_rows)                     # every row weighs the same

    if config.coding not in ("ls", "group_lasso"):
        raise ValueError(f"coding must be 'ls' or 'group_lasso', got {config.coding!r}")
    # one group per (perturbation, context): the rows that share a selection
    # of programs under the group lasso
    from .models.group_lasso import group_batches, group_penalty, lasso_groups
    from .models.loadings import collinearity_hinge, gene_concentration, gene_prior
    gkey = lasso_groups(perturbation_idx, context_idx, config.lasso_groups)
    gkey_t = torch.from_numpy(gkey).to(device)
    if config.coding == "group_lasso":
        model.loadings.renormalize_rows_()
    X_t = torch.from_numpy(X).to(device)
    w_t = torch.from_numpy(w.astype(np.float32)).to(device)
    p_all = torch.from_numpy(perturbation_idx).to(device)
    c_all = torch.from_numpy(context_idx).to(device)
    if target_col is None or config.no_mask:
        col_t = None
    else:
        col_t = torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)

    if config.label_weight_decay:
        label_params = set(map(id, model.label_net.parameters()))
        optimiser = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in label_params],
             "weight_decay": 0.0},
            {"params": list(model.label_net.parameters()),
             "weight_decay": config.label_weight_decay},
        ], lr=config.lr)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)

    history: list[dict] = []
    best = {"score": None, "epoch": -1, "state": None}

    for epoch in range(config.epochs):
        model.train()
        # drawn every epoch, also when the group lasso builds its own batches,
        # so the random stream does not depend on the coding
        order = rng.permutation(len(train_idx))
        totals = {"disc": 0.0, "recon": 0.0, "accuracy": 0.0}
        if config.coding == "group_lasso":
            totals.update(lasso_penalty=0.0, programs_per_pair=0.0)
        totals.update(gene_penalty=0.0, gene_top_share=0.0, distinct_penalty=0.0)
        n_batches = 0
        if config.coding == "group_lasso":
            # batches of whole pairs, so each pair's rows are solved together
            batches = group_batches(gkey[train_idx], config.batch_size, rng)
        else:
            batches = [order[s:s + config.batch_size] for s in range(0, len(train_idx), config.batch_size)]

        for local in batches:
            if len(local) < 2:
                continue
            rows = train_idx[local]          # positions in X
            rows_t = torch.from_numpy(rows).to(device)

            x = X_t[rows_t]
            cols = col_t[rows_t] if col_t is not None else None
            neg_p, neg_c = sampler.sample(local, rng)

            z = model.project(x, cols, gkey_t[rows_t])
            # the reconstruction term uses the raw lasso activities (the lasso
            # objective); the discriminator uses z
            z_rec = model.lasso_raw_ if config.coding == "group_lasso" else z
            neg_p_t = torch.from_numpy(neg_p).to(device)
            neg_c_t = torch.from_numpy(neg_c).to(device)

            # ---- L_disc: each row under its own label (true) and under a
            # swapped label (false), one of each per row
            pos = model.score(z, p_all[rows_t], c_all[rows_t])
            neg = model.score(z, neg_p_t, neg_c_t)
            loss_d = contrastive_loss(pos, neg)

            # ---- L_lasso: masked reconstruction error + group penalty
            per_row = model.loadings.squared_error(x, z_rec, cols)
            loss_r = weighted_squared_error(per_row, w_t[rows_t])
            if config.coding == "group_lasso":
                # the lasso objective 1/2 |r|^2 + lam * penalty, times 2 / (n G)
                # to match the error term (a mean over rows of |r|^2 / G)
                gk = gkey_t[rows_t]
                pen = 2.0 * group_penalty(z_rec, gk, config.lasso_lambda) / (len(rows) * x.shape[1])
                loss_r = loss_r + pen
                totals["lasso_penalty"] += float(pen)
                sel = model.lasso_sel_
                first = torch.ones_like(gk, dtype=torch.bool)
                srt, idx = torch.sort(gk)
                first[idx[1:]] = srt[1:] != srt[:-1]
                totals["programs_per_pair"] += float(sel[first].sum(1).float().mean())

            loss = loss_d + config.alpha * loss_r

            # ---- gene-side prior on the programs: few genes per program, and
            # programs kept distinct (no near-collinear pairs)
            Bcur = model.loadings.B
            share = gene_concentration(Bcur, 200)
            loss_g = gene_prior(Bcur)
            if config.gene_sparsity:
                loss = loss + config.gene_sparsity * loss_g
            loss_c = collinearity_hinge(Bcur, config.distinct_tau)
            if config.distinct_weight:
                loss = loss + config.distinct_weight * loss_c
            totals["gene_penalty"] += float(loss_g)
            totals["gene_top_share"] += float(share.mean())
            totals["distinct_penalty"] += float(loss_c)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            if config.coding == "group_lasso":
                model.loadings.renormalize_rows_()

            totals["disc"] += float(loss_d)
            totals["recon"] += float(loss_r)
            totals["accuracy"] += discrimination_accuracy(pos, neg)
            n_batches += 1

        record = {
            "epoch": epoch,
            **{k: v / max(n_batches, 1) for k, v in totals.items()},
        }
        # the quantity actually minimised (recon holds the lasso penalty under
        # the group lasso)
        record["total"] = (record["disc"] + config.alpha * record["recon"]
                           + config.gene_sparsity * record["gene_penalty"]
                           + config.distinct_weight * record["distinct_penalty"])
        is_last = epoch == config.epochs - 1
        evaluated = False
        if val_idx is not None and config.eval_every and (
            epoch % config.eval_every == 0 or is_last
        ):
            evaluated = True
            record.update(
                {
                    f"val_{k}": v
                    for k, v in evaluate(
                        model,
                        X_t,
                        perturbation_idx,
                        context_idx,
                        val_idx,
                        eval_sampler,
                        np.random.default_rng(config.seed + 10_000 + epoch),
                        col_t,
                        w_t,
                    ).items()
                }
            )
            # the training objective on the validation rows, term by term
            with torch.no_grad():
                Bv = model.loadings.B
                record["val_gene_penalty"] = float(gene_prior(Bv))
                record["val_distinct_penalty"] = float(collinearity_hinge(Bv, config.distinct_tau))
                record["val_gene_top_share"] = float(gene_concentration(Bv, 200).mean())
                Bu = Bv / Bv.norm(dim=1, keepdim=True)
                record["val_max_program_cos"] = float(((Bu @ Bu.T).abs() - torch.eye(len(Bu), device=Bu.device)).max())
            record["val_total"] = (record["val_disc_objective"]
                                   + config.alpha * record.get("val_lasso_objective", record["val_recon"])
                                   + config.gene_sparsity * record["val_gene_penalty"]
                                   + config.distinct_weight * record["val_distinct_penalty"])
            score = record["val_auc"]
            improved = best["score"] is None or score > best["score"]
            if improved:
                best = {
                    "score": score,
                    "epoch": epoch,
                    "state": {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    },
                }
            record["val_best"] = improved
            if on_eval is not None:
                on_eval(epoch, model)

        history.append(record)
        if config.log_every and (epoch % config.log_every == 0 or is_last or evaluated):
            line = (
                f"epoch {epoch:4d}  disc {record['disc']:.4f}  "
                f"recon {record['recon']:.4f}  acc {record['accuracy']:.3f}"
            )
            if "val_accuracy" in record:
                line += (
                    f"   | val auc {record['val_auc']:.3f}  acc {record['val_accuracy']:.3f}  "
                    f"recon {record['val_recon']:.4f}  "
                    f"r {record['val_span_residual_median']:.3f}"
                    f"{'  *best' if record.get('val_best') else ''}"
                )
            print(line, flush=True)

    # the last epoch's parameters, kept before the best ones are restored
    model.final_state_ = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.final_epoch_ = len(history) - 1
    model.selected_epoch_ = best["epoch"]
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(
            f"restored parameters from epoch {best['epoch']} "
            f"(val auc {best['score']:.4f})",
            flush=True,
        )
        for rec in history:
            rec["selected_epoch"] = best["epoch"]

    return model, history


@torch.no_grad()
def evaluate(
    model: Extract,
    X_t: torch.Tensor,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    rows: np.ndarray,
    sampler: StratifiedNegativeSampler,
    rng: np.random.Generator,
    target_col: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    batch_size: int = 2048,
) -> dict:
    """Metrics on held-out rows, scored exactly as training scores them.

    Each row is scored under its own label (a positive) and under one swapped
    label drawn by ``sampler`` (a negative).

    ``auc`` / ``accuracy`` / ``disc``
        True vs swapped labels. Both ``p`` and ``c`` of a validation pair were
        seen in training but *never together*, so this is the interaction test:
        **a lookup table scores at chance here**. ``auc`` (rank-based) picks the
        epoch.
    ``disc_objective``
        The discriminator loss as training minimises it: mean BCE of the
        positives + mean BCE of the negatives.
    ``recon``
        Masked squared error per gene of the activities the discriminator sees.
    ``lasso_objective`` / ``programs_per_pair``
        Under the group lasso: the reconstruction term as training minimises it
        (raw lasso activities, plus the penalty), and how many programs a pair
        selects.
    ``span_residual_median`` / ``_q90``
        ``||x - x B^+ B|| / ||x||``: do the held-out pairs need programs outside
        ``span(B)``?
    ``real_score_mean`` / ``fake_score_mean``
        The raw logits; a collapsed head shows up as both near zero.
    """
    was_training = model.training
    model.eval()
    # Take the device from the MODEL, not from X_t: a caller may keep the full
    # matrix on the CPU on purpose and let batches move.
    device = next(model.parameters()).device
    src = X_t.device
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and (rows.max() >= len(X_t) or rows.min() < 0):
        raise IndexError(
            f"rows span [{rows.min()}, {rows.max()}] but X has {len(X_t)} rows"
        )
    if len(rows) != len(sampler.perturbation_idx):
        raise ValueError(
            f"sampler was built on {len(sampler.perturbation_idx)} rows but "
            f"{len(rows)} were passed; sample() indexes the sampler's own "
            "rows, so the two must correspond one-to-one"
        )
    if not len(rows):
        raise ValueError("no rows to evaluate")

    def _bce_sum(logits: torch.Tensor, target: float) -> float:
        return float(F.binary_cross_entropy_with_logits(
            logits, torch.full_like(logits, target), reduction="sum"))

    # Accumulated as sums and counts, so every row weighs the same.
    t = dict.fromkeys(["pos_ok", "pos_bce", "pos_sum", "perm_ok", "perm_bce", "perm_sum",
                       "wsq", "w"], 0.0)
    n_pos = 0
    residuals = []
    pos_all, perm_all = [], []
    lasso = model.config.coding == "group_lasso"
    if lasso:
        from .models.group_lasso import group_batches, group_penalty, lasso_groups
        gkey = lasso_groups(perturbation_idx[rows], context_idx[rows], model.config.lasso_groups)
        batches = group_batches(gkey, batch_size)
        t.update(lasso_raw_sq=0.0, lasso_pen=0.0, n_groups=0.0, n_programs=0.0)
    else:
        batches = [np.arange(s, min(s + batch_size, len(rows)))
                   for s in range(0, len(rows), batch_size)]

    for local in batches:
        idx = rows[local]
        idx_t = torch.from_numpy(idx).to(src)
        x = X_t[idx_t].to(device)
        cols = (target_col[idx_t].to(device)
                if target_col is not None else None)
        z = model.project(x, cols, torch.from_numpy(gkey[local]).to(device)
                          if lasso else None)
        if lasso:
            g_t = torch.from_numpy(gkey[local]).to(device)
            t["lasso_raw_sq"] += float(model.loadings.squared_error(x, model.lasso_raw_, cols).sum())
            t["lasso_pen"] += float(group_penalty(model.lasso_raw_, g_t, model.config.lasso_lambda))
            _, first = np.unique(gkey[local], return_index=True)
            t["n_groups"] += len(first)
            t["n_programs"] += float(model.lasso_sel_[torch.from_numpy(first).to(device)].sum())
        pos = model.score(
            z,
            torch.from_numpy(perturbation_idx[idx]).to(device),
            torch.from_numpy(context_idx[idx]).to(device),
        )
        neg_p, neg_c = sampler.sample(local, rng)
        perm = model.score(
            z,
            torch.from_numpy(neg_p).to(device),
            torch.from_numpy(neg_c).to(device),
        )
        t["pos_ok"] += float((pos > 0).sum())
        t["pos_bce"] += _bce_sum(pos, 1.0)
        t["pos_sum"] += float(pos.sum())
        t["perm_ok"] += float((perm <= 0).sum())
        t["perm_bce"] += _bce_sum(perm, 0.0)
        t["perm_sum"] += float(perm.sum())
        pos_all.append(pos.cpu().numpy()); perm_all.append(perm.cpu().numpy())

        per_row = model.loadings.squared_error(x, z, cols)
        w = (row_weights[idx_t].to(device) if row_weights is not None
             else torch.ones_like(per_row))
        t["wsq"] += float((per_row * w).sum())
        t["w"] += float(w.sum())
        residuals.append(model.loadings.span_residual(x, cols).cpu().numpy())
        n_pos += len(idx)

    model.train(was_training)
    r = np.concatenate(residuals)
    from sklearn.metrics import roc_auc_score
    P, N = np.concatenate(pos_all), np.concatenate(perm_all)
    out = {"auc": float(roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(N))], np.r_[P, N]))}
    if lasso:
        n_genes = X_t.shape[1]
        out["programs_per_pair"] = t["n_programs"] / max(t["n_groups"], 1)
        # the lasso objective per row and gene, as in training (squared_error
        # is the per-row mean over genes)
        out["lasso_objective"] = (t["lasso_raw_sq"] + 2.0 * t["lasso_pen"] / n_genes) / max(n_pos, 1)
    return {
        **out,
        "n_rows": int(n_pos),
        "accuracy": (t["pos_ok"] + t["perm_ok"]) / (2 * n_pos),
        "disc": (t["pos_bce"] + t["perm_bce"]) / n_pos,
        "disc_objective": t["pos_bce"] / n_pos + t["perm_bce"] / n_pos,
        "recon": t["wsq"] / max(t["w"], 1e-12),
        "real_score_mean": t["pos_sum"] / n_pos,
        "fake_score_mean": t["perm_sum"] / n_pos,
        "span_residual_median": float(np.median(r)),
        "span_residual_q90": float(np.quantile(r, 0.90)),
    }


# ---------------------------------------------------------------------------


def prepare(
    X: np.ndarray,
    meta,
    genes: np.ndarray,
    mask_on_target: bool = True,
    perturbation_levels: np.ndarray | None = None,
    context_levels: np.ndarray | None = None,
) -> dict:
    """Turn ``(X, meta)`` from :func:`extract.de.load_matrices` into
    :func:`fit` arguments.

    ``meta`` needs ``perturbation`` and ``context`` (and ``label`` only through
    the caller's choice of rows). Cell counts and subsample indices are not
    passed on: the model sees the matrix and the labels only.

    Pass ``perturbation_levels`` / ``context_levels`` from a previous call to
    encode a second row set (the test partition) with the **same** indices, so
    its labels address the embeddings the model was trained with. A label
    absent from the given levels is an error, not a new index.
    """
    import pandas as pd

    perturbations = meta["perturbation"].to_numpy()
    contexts = meta["context"].to_numpy()
    labels = meta["label"].astype(str).to_numpy()

    def _encode(values, levels, what):
        if levels is None:
            return pd.factorize(values, sort=True)
        codes = pd.Categorical(values, categories=levels).codes
        if (codes < 0).any():
            unknown = sorted(set(values[codes < 0]))
            raise ValueError(
                f"{len(unknown)} {what} not in the given levels, e.g. {unknown[:3]}"
            )
        return codes, np.asarray(levels)

    pert_codes, pert_levels = _encode(perturbations, perturbation_levels, "perturbations")
    ctx_codes, ctx_levels = _encode(contexts, context_levels, "contexts")

    out = {
        "X": np.asarray(X, dtype=np.float32),
        "perturbation_idx": pert_codes.astype(np.int64),
        "context_idx": ctx_codes.astype(np.int64),
        "target_col": (
            on_target_index(perturbations, genes) if mask_on_target else None
        ),
    }
    out["perturbation_levels"] = np.asarray(pert_levels)
    out["context_levels"] = np.asarray(ctx_levels)
    return out


@torch.no_grad()
def encode(
    model: Extract, X: np.ndarray, target_col: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Factor activations ``Z`` for all rows of ``X`` (``groups``: the rows'
    group ids, needed with ``coding="group_lasso"``; see ``lasso_groups``)."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    cols = (
        torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)
        if target_col is not None
        else None
    )
    g = torch.from_numpy(np.asarray(groups, dtype=np.int64)).to(device) if groups is not None else None
    return model.project(x, cols, g).cpu().numpy()


@torch.no_grad()
def reconstruct(
    model: Extract, X: np.ndarray, target_col: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """``X_hat = z B`` with ``z`` the activities of ``X`` (as :func:`encode`)."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    cols = (
        torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)
        if target_col is not None
        else None
    )
    g = torch.from_numpy(np.asarray(groups, dtype=np.int64)).to(device) if groups is not None else None
    return model.loadings.reconstruct(model.project(x, cols, g)).cpu().numpy()


__all__ = [
    "Extract",
    "evaluate",
    "TrainConfig",
    "fit",
    "prepare",
    "encode",
    "reconstruct",
    "NO_MASK",
]


def load_model(sweep_dir, tag: str, which: str = "best", device: str = "cpu") -> Extract:
    """Rebuild a model saved by ``scripts/sweep_grid.py`` from
    ``config_<tag>.json`` and ``model_<which>_<tag>.pt`` (``which`` is
    ``"best"``, the selected epoch, or ``"final"``, the last one). The fixed
    subspace V is part of the saved state, so nothing else is needed; score
    rows with ``model.project`` / ``model.score`` after encoding their labels
    with ``levels_<tag>.npy`` (perturbation and context levels)."""
    import dataclasses, json
    from pathlib import Path
    sweep_dir = Path(sweep_dir)
    cfg_d = json.loads((sweep_dir / f"config_{tag}.json").read_text())
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    kw = {k: (tuple(v) if k == "basis" else v) for k, v in cfg_d.items() if k in names}
    kw.setdefault("negative_draw", "labels")     # saved before the option: uniform draws
    kw.setdefault("lasso_groups", "pair")        # saved before the option: the group lasso
    kw["device"] = device
    config = TrainConfig(**kw)
    state = torch.load(sweep_dir / f"model_{which}_{tag}.pt", map_location=device)
    # any basis of the right shape will do: the saved state overwrites it
    basis = (state["loadings.V"] if "loadings.V" in state else state["loadings.B"]).numpy()
    model = Extract(n_genes=cfg_d["n_genes"], n_perturbations=cfg_d["n_perturbations"],
                    n_contexts=cfg_d["n_contexts"], config=config, subspace_basis=basis)
    model.load_state_dict(state)
    return model.to(device).eval()
