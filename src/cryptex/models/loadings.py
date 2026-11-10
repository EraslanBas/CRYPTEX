"""The one factor->gene map, the projection derived from it, and the gene-side prior.

    B      [d, G]   the only interpretable output, the only learned map (B = A V)
    B^+    [G, d]   B^T (B B^T)^{-1}  -- computed, never fitted
    z_ls   [d]      argmin_z ||m * (x - z B)||^2, the start of the group lasso
                    and the projection behind the span residual

There is deliberately no encoder. With a separate ``W`` there are two maps
between factors and genes, they are free to disagree about what factor ``k``
means, and only one of them is a loading vector: encoder weights are *filters*,
not *patterns*, and a filter can place large weight on a gene precisely to
cancel it as a nuisance (Haufe et al. 2014). Tying removes the question --
``B`` is a pattern by construction.

``x`` is therefore never on a generative path. Its activities are fitted
through ``B`` (``models.group_lasso``) so the discriminator can judge it, and
it is the regression target.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Sentinel in a ``target_col`` array: this row masks nothing (the perturbation
#: targets no measured gene, e.g. a non-targeting control).
NO_MASK = -1


class GlobalLoadings(nn.Module):
    """``B`` [n_factors, n_genes], plus the projection and reconstruction.

    Parameters
    ----------
    ridge
        Added to the diagonal of ``B B^T`` before inversion. Small but not
        optional: early in training rows of ``B`` are near-collinear and the
        Gram matrix is ill-conditioned.
    """

    def __init__(self, n_factors: int, n_genes: int, ridge: float = 1e-4):
        super().__init__()
        self.n_factors = n_factors
        self.n_genes = n_genes
        self.ridge = ridge
        self.B = nn.Parameter(torch.empty(n_factors, n_genes))
        nn.init.xavier_uniform_(self.B)

    # ---- the Gram matrix and its inverse (one d x d factorisation per step) --

    def gram_inverse(self) -> torch.Tensor:
        """``(B B^T + ridge I)^{-1}`` -- [d, d]. Cheap: ``d`` is 20-30."""
        G = self.B @ self.B.T
        G = G + self.ridge * torch.eye(
            self.n_factors, device=G.device, dtype=G.dtype
        )
        return torch.linalg.inv(G)

    @property
    def pinv(self) -> torch.Tensor:
        """``B^+ = B^T (B B^T)^{-1}`` -- [n_genes, n_factors]."""
        return self.B.T @ self.gram_inverse()

    # ---- projection -------------------------------------------------------

    def project(
        self,
        x: torch.Tensor,
        target_col: torch.Tensor | None = None,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Masked least-squares coordinates of ``x`` in ``B``'s row space.

        Unmasked this is ``z = x B^+``. With one masked entry per row it is a
        rank-one downdate of the shared Gram matrix, applied via
        Sherman-Morrison so that one ``d x d`` inverse serves every
        row and each row costs ``O(d^2)`` however many distinct genes the batch
        masks.

        Parameters
        ----------
        x
            [batch, n_genes].
        target_col
            [batch] int64 column to exclude per row, :data:`NO_MASK` for none.
            ``None`` masks nothing at all.
        tol
            Rows whose Sherman-Morrison denominator ``1 - b^T G^-1 b`` falls
            below this are re-solved explicitly. The identity is exact but
            loses precision as the denominator approaches zero, which happens
            when a masked gene's loading column nearly spans a factor
            direction on its own.
        """
        Ginv = self.gram_inverse()                      # [d, d], symmetric
        if target_col is None:
            return (x @ self.B.T) @ Ginv                # z = x B^+

        masked = target_col != NO_MASK
        x_tilde = x
        if masked.any():
            x_tilde = x.clone()
            rows = torch.nonzero(masked, as_tuple=True)[0]
            x_tilde[rows, target_col[rows]] = 0.0

        Bx = x_tilde @ self.B.T                         # [n, d]
        z = Bx @ Ginv
        if not masked.any():
            return z

        # Sherman-Morrison correction on the masked rows only.
        b = self.B[:, target_col.clamp(min=0)].T        # [n, d]; junk where unmasked
        Ginv_b = b @ Ginv                               # [n, d]
        denom = 1.0 - (Ginv_b * b).sum(-1)              # [n]
        num = (Ginv_b * Bx).sum(-1)                     # [n]

        stable = masked & (denom.abs() > tol)
        correction = torch.zeros_like(z)
        if stable.any():
            correction[stable] = (
                Ginv_b[stable] * (num[stable] / denom[stable]).unsqueeze(-1)
            )
        z = z + correction                              # Sherman-Morrison

        # Explicit solve for the few rows where the identity is ill-conditioned.
        unstable = masked & ~stable
        if unstable.any():
            idx = torch.nonzero(unstable, as_tuple=True)[0]
            bb = b[idx]                                 # [k, d]
            # Rebuild the Gram matrix directly rather than inverting Ginv: this
            # is the path chosen precisely because the system is ill-conditioned,
            # so inverting an inverse is the last thing to do here.
            G_full = self.B @ self.B.T + self.ridge * torch.eye(
                self.n_factors, device=self.B.device, dtype=self.B.dtype
            )
            G = G_full.unsqueeze(0) - bb.unsqueeze(2) * bb.unsqueeze(1)
            z = z.clone()
            z[idx] = torch.linalg.solve(G, Bx[idx].unsqueeze(-1)).squeeze(-1)
        return z

    @torch.no_grad()
    def renormalize_rows_(self) -> None:
        """Scale each program's loadings to unit length (in place). Needed when
        the activities are penalised (group lasso): otherwise larger loadings
        would shrink the activities and the penalty with no change to the fit."""
        self.B.div_(self.B.norm(dim=1, keepdim=True).clamp_min(1e-12))

    # ---- reconstruction ---------------------------------------------------

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """``x_hat = z B`` -- [batch, n_genes]. No bias: ``x`` is already a
        difference against within-context controls, so zero is the reference."""
        return z @ self.B

    def squared_error(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        target_col: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-row ``|| m * (x - z B) ||^2 / n_retained`` -- [batch].

        Divided by the number of retained genes so the scale does not depend on
        ``G`` or on whether a row masks anything.
        """
        resid = x - self.reconstruct(z)
        sq = resid.pow(2).sum(dim=1)
        n_kept = torch.full_like(sq, float(self.n_genes))
        if target_col is not None:
            masked = target_col != NO_MASK
            if masked.any():
                rows = torch.nonzero(masked, as_tuple=True)[0]
                dropped = resid[rows, target_col[rows]].pow(2)
                sq = sq.index_add(0, rows, -dropped)
                n_kept = n_kept.index_add(0, rows, -torch.ones_like(dropped))
        return sq / n_kept

    # ---- diagnostics ------------------------------------------------------

    @torch.no_grad()
    def span_residual(
        self, x: torch.Tensor, target_col: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``||m * (x - x B^+ B)|| / ||m * x||`` per row (the span residual).

        The one measurement that can invalidate the architecture rather than
        call for a tuning pass: a held-out drug whose rows carry a large
        orthogonal residual introduced a program the training contexts never
        showed, and no amount of nonlinearity elsewhere can absorb it.

        Pass ``target_col`` so the on-target entry is excluded from both the
        projection and the norms, exactly as in training. ``B`` is fitted never
        to reconstruct that entry, and it carries a median ~25% of a row's
        squared norm, so leaving it in charges ``B`` for mass it was told to
        ignore and inflates ``r``. Omitting ``target_col`` reproduces the
        earlier unmasked behaviour.
        """
        z = self.project(x, target_col)
        resid = x - self.reconstruct(z)
        if target_col is not None:
            masked = target_col != NO_MASK
            if masked.any():
                rows = torch.nonzero(masked, as_tuple=True)[0]
                resid = resid.clone()
                x = x.clone()
                resid[rows, target_col[rows]] = 0.0
                x[rows, target_col[rows]] = 0.0
        num = torch.linalg.norm(resid, dim=1)
        return num / torch.linalg.norm(x, dim=1).clamp_min(1e-12)


def gene_concentration(B: torch.Tensor, k: int = 200) -> torch.Tensor:
    """[n_factors] share of each program's squared loading in its ``k``
    largest-|loading| genes (1 = the program lives entirely in k genes).
    The top set is re-chosen from the current B; gradients flow through the
    loadings' values. Raw loadings, no gene scaling."""
    sq = B * B
    return torch.topk(sq, min(k, sq.shape[1]), dim=1).values.sum(1) / sq.sum(1).clamp_min(1e-20)


def l1_sparseness(B: torch.Tensor) -> torch.Tensor:
    """[n_factors] ``||b_k||_1 / (||b_k||_2 sqrt(G))`` per program: 1 for a flat
    profile, ``1/sqrt(G)`` for a single gene (the Hoyer-type sparseness
    measure; lower = heavier-tailed loadings). Scale-free; raw loadings."""
    return B.abs().sum(1) / (B.norm(dim=1).clamp_min(1e-20) * B.shape[1] ** 0.5)


def collinearity_hinge(B: torch.Tensor, tau: float) -> torch.Tensor:
    """``sum_{k<l} max(0, |cos(b_k, b_l)| - tau)^2``: zero unless two programs
    point in nearly the same direction (shared genes are free below ``tau``)."""
    Bu = B / B.norm(dim=1, keepdim=True).clamp_min(1e-20)
    C = (Bu @ Bu.T).abs()
    iu = torch.triu_indices(C.shape[0], C.shape[0], 1, device=C.device)
    return (torch.relu(C[iu[0], iu[1]] - tau) ** 2).sum()


def gene_prior(B: torch.Tensor) -> torch.Tensor:
    """The gene-side penalty: sum_k l1_sparseness(b_k), heavy-tailed loadings."""
    return l1_sparseness(B).sum()


def initial_axes(d: int, axes_init: str = "jd", seed: int = 0) -> torch.Tensor:
    """Starting ``A`` [d, d] for ``B = A V``: a random rotation drawn from
    ``seed`` (different for every seed, so that cross-seed agreement cannot come
    from a shared starting point), or -- for ``"jd"``, whose axes are computed
    from the data (:func:`cryptex.train.jd_axes`) and passed in -- the identity
    as a placeholder that ``load_state_dict`` overwrites when a saved model is
    rebuilt."""
    if axes_init == "jd":
        return torch.eye(d)
    if axes_init == "random":
        Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
        return torch.as_tensor(Q, dtype=torch.float32)
    raise ValueError(f"axes_init must be 'jd' or 'random', got {axes_init!r}")


class FixedBasisLoadings(GlobalLoadings):
    """``B = A V`` with ``V`` [d, G] fixed and only ``A`` [d, d] learned.

    The span of ``B`` is the span of ``V`` and cannot move, so reconstruction is
    fixed by ``V`` and the discriminator chooses only the axes inside it --
    oblique ones, since ``A`` is any invertible matrix: the structure of ICA
    (whiten, then unmix) with the separable head as the unmixing criterion.
    Every method of :class:`GlobalLoadings` works unchanged through ``B``.
    """

    def __init__(self, V: np.ndarray, ridge: float = 1e-4,
                 axes_init: str = "jd", seed: int = 0, A0: torch.Tensor | None = None):
        nn.Module.__init__(self)
        V = torch.as_tensor(np.asarray(V), dtype=torch.float32)
        self.n_factors, self.n_genes = V.shape
        self.ridge = ridge
        self.register_buffer("V", V)
        if A0 is None:
            A0 = initial_axes(self.n_factors, axes_init, seed)
        self.A = nn.Parameter(A0)

    @torch.no_grad()
    def renormalize_rows_(self) -> None:
        """Unit-length rows of ``B = A V``, by rescaling the rows of ``A``."""
        self.A.div_((self.A @ self.V).norm(dim=1, keepdim=True).clamp_min(1e-12))

    @property
    def B(self) -> torch.Tensor:  # type: ignore[override]
        return self.A @ self.V
