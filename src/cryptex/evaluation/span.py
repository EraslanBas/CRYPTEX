"""The span test -- eq. (14). The one result that can invalidate the design.

    r(x) = ||x - x B^+ B|| / ||x||

A global ``B`` forbids something narrow and sharp: an interaction may only
re-weight the same ``d`` programs. A drug x perturbation combination that
switches on a program appearing in no single condition falls outside
``span(B)`` and cannot be represented, however nonlinear anything else is.

Contexts, not perturbations, are the scarce resource. ``B`` is learned from
logFC measured against within-context controls, so each drug's own main effect
is differenced out and what spans ``B`` is "what knocking genes down does,
across the drugs in training". A program that only surfaces under a held-out
drug has no route into the span however many perturbations are available: 2,050
perturbations pin down each program's loadings, but 16 drugs are all there is to
establish which programs exist.

So measure it rather than betting on it. Small residuals mean re-weighting is
enough; a drug whose rows carry a large residual introduced a program the
training contexts never showed -- the one failure mode a tuning pass cannot fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpanResidual:
    context: str
    n_rows: int
    median: float
    q90: float
    #: Residual of the same rows against a PCA basis of the same rank fitted on
    #: the same training contexts. The comparison that matters: a large residual
    #: is only evidence against *this* B if an unconstrained rank-d basis does
    #: better on the same held-out rows.
    median_pca: float | None = None

    def __str__(self) -> str:
        base = (
            f"{self.context:<16} n={self.n_rows:>6}  "
            f"median r={self.median:.3f}  q90={self.q90:.3f}"
        )
        if self.median_pca is not None:
            base += f"  (rank-matched PCA {self.median_pca:.3f})"
        return base


def span_residual(B: np.ndarray, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    """``r(x)`` per row of ``X`` for loadings ``B`` [d, G]."""
    Bt = torch.from_numpy(np.asarray(B, dtype=np.float64))
    G = Bt @ Bt.T
    G = G + 1e-10 * torch.eye(G.shape[0], dtype=G.dtype)
    Ginv = torch.linalg.inv(G)

    out = np.empty(len(X))
    for start in range(0, len(X), batch):
        x = torch.from_numpy(np.asarray(X[start : start + batch], dtype=np.float64))
        z = (x @ Bt.T) @ Ginv
        resid = torch.linalg.norm(x - z @ Bt, dim=1)
        norm = torch.linalg.norm(x, dim=1).clamp_min(1e-12)
        out[start : start + batch] = (resid / norm).numpy()
    return out


def leave_one_context_out(
    B_by_heldout: dict[str, np.ndarray],
    X: np.ndarray,
    contexts: np.ndarray,
    compare_pca: bool = True,
) -> list[SpanResidual]:
    """Span residual of each held-out context under a ``B`` fitted without it.

    Parameters
    ----------
    B_by_heldout
        ``{held_out_context: B}``, each ``B`` fitted on the *other* contexts.
    X, contexts
        The full row set and each row's context.

    Notes
    -----
    Report this alongside the rank-matched PCA baseline. The absolute residual
    of a 24-dimensional basis on an 18,154-dimensional row is large by
    construction; what carries information is whether *this* ``B`` is worse than
    an unconstrained basis of the same rank on the same rows.
    """
    contexts = np.asarray(contexts)
    results = []
    for ctx, B in B_by_heldout.items():
        rows = contexts == ctx
        if not rows.any():
            continue
        r = span_residual(B, X[rows])
        median_pca = None
        if compare_pca:
            train = X[~rows]
            median_pca = float(
                np.median(span_residual(_pca_basis(train, B.shape[0]), X[rows]))
            )
        results.append(
            SpanResidual(
                context=str(ctx),
                n_rows=int(rows.sum()),
                median=float(np.median(r)),
                q90=float(np.quantile(r, 0.90)),
                median_pca=median_pca,
            )
        )
    return results


def _pca_basis(X: np.ndarray, d: int) -> np.ndarray:
    """Top-``d`` right singular vectors of ``X`` -- the rank-matched baseline."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d]
