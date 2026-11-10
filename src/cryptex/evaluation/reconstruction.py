"""How well a set of gene loadings reconstructs held-out responses.

One function, used for CRYPTEX and for every baseline, so the numbers are
comparable: each row is projected onto ``span(B)`` by masked least squares (the
on-target entry excluded, exactly as CRYPTEX projects), reconstructed as
``z B``, and scored on the retained entries. Zero is the reference -- rows are
log-fold changes against within-context controls, so no mean is added back.

Reconstruction depends on ``span(B)`` only, not on the axes inside it, so these
numbers compare subspaces; they cannot tell two rotations of the same subspace
apart.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.loadings import NO_MASK, GlobalLoadings


def reconstruct(B: np.ndarray, X: np.ndarray, target_col: np.ndarray | None = None,
                batch: int = 8192) -> np.ndarray:
    """``z B`` with ``z`` the masked least-squares projection of each row of ``X``."""
    B = np.asarray(B, dtype=np.float32)
    gl = GlobalLoadings(B.shape[0], B.shape[1], ridge=1e-6)
    cols = (np.full(len(X), NO_MASK) if target_col is None
            else np.asarray(target_col, dtype=np.int64))
    out = np.empty(X.shape, dtype=np.float32)
    with torch.no_grad():
        gl.B.copy_(torch.from_numpy(B))
        for i in range(0, len(X), batch):
            x = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32))
            c = torch.from_numpy(cols[i:i + batch])
            out[i:i + batch] = gl.reconstruct(gl.project(x, c)).numpy()
    return out


def reconstruction_metrics(
    B: np.ndarray,
    X: np.ndarray,
    target_col: np.ndarray | None = None,
    stratum: np.ndarray | None = None,
    threshold: float = float(np.log(1.2)),
) -> dict:
    """Reconstruction of ``X`` [rows, G] from loadings ``B`` [d, G].

    ``R2_*`` are fractions of squared norm reconstructed (uncentred); ``slope``
    is the regression of reconstructed on true entries; the ``affected_*``
    numbers are over entries with ``|x| > threshold``. ``stratum`` (0 = the
    full-data row) splits the per-row number by row type.
    """
    X = np.asarray(X, dtype=np.float32)
    Xh = reconstruct(B, X, target_col)
    M = np.ones(X.shape, dtype=bool)
    if target_col is not None:
        r = np.nonzero(np.asarray(target_col) != NO_MASK)[0]
        M[r, np.asarray(target_col)[r]] = False
    R = (X - Xh) * M
    sst_row = (X * X * M).sum(1)
    row_r2 = 1 - (R * R).sum(1) / np.maximum(sst_row, 1e-12)
    sst_gene = (X * X * M).sum(0)
    gene_r2 = 1 - (R * R).sum(0) / np.maximum(sst_gene, 1e-12)
    x, xh = X[M], Xh[M]
    big = np.abs(x) > threshold
    out = {
        "d": int(B.shape[0]),
        "R2_global": 1 - float((R * R).sum()) / float(sst_row.sum()),
        "R2_row_median": float(np.median(row_r2)),
        "R2_gene_median": float(np.median(gene_r2)),
        "pearson_entries": float(np.corrcoef(x, xh)[0, 1]),
        "slope": float((x * xh).sum() / (x * x).sum()),
        "affected_sign_agreement": float((np.sign(xh[big]) == np.sign(x[big])).mean()),
        "affected_recall": float((np.abs(xh[big]) > threshold).mean()),
        "affected_precision": float((np.abs(x[np.abs(xh) > threshold]) > threshold).mean()),
        "span_residual_median": float(np.median(
            np.sqrt((R * R).sum(1)) / np.maximum(np.sqrt(sst_row), 1e-12))),
    }
    if stratum is not None:
        st = np.asarray(stratum)
        out["R2_row_median_full"] = float(np.median(row_r2[st == 0]))
        out["R2_row_median_subsample"] = float(np.median(row_r2[st > 0]))
    return out
