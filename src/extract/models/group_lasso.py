"""Lasso activities: one selection of programs per group of rows.

By default every row is its own group (``lasso_groups(..., "row")``): a lasso
per row, ``1/2 ||m * (x - z B)||^2 + lam * sum_k |z_k|``, so a row's
activities depend on that row alone. With the rows of a (perturbation,
context) pair as one group it is the group lasso below (kept for comparison).

For a group g of rows (all rows of one (perturbation, context) pair: the
full-data estimate and its pseudobulks), the activities Z_g [n_g, d] solve

    min_Z  1/2 sum_rows || m_r * (x_r - z_r B) ||^2  +  lam * sqrt(n_g) * sum_k || Z_g[:, k] ||_2

``m_r`` masks a row's on-target gene. The penalty is the length of each
program's column within the group, so a program is used by all rows of the
pair or by none; its activity can still differ between the pair's rows.
``sqrt(n_g)`` keeps the penalty comparable between pairs with few and many
rows. Solved by FISTA (proximal gradient with momentum), unrolled for a fixed
number of steps so gradients flow back to B; started at the least-squares
activities. The same activities are used everywhere: in the reconstruction
objective and by the discriminator.
"""

from __future__ import annotations

import torch

NO_MASK = -1


def _mask(x: torch.Tensor, target_col: torch.Tensor | None) -> torch.Tensor:
    m = torch.ones_like(x)
    if target_col is not None:
        r = torch.nonzero(target_col != NO_MASK, as_tuple=True)[0]
        m[r, target_col[r]] = 0.0
    return m


def _group_prox(Z, groups, n_groups, thresh):
    """Block soft-thresholding of each (group, program) column."""
    sq = torch.zeros(n_groups, Z.shape[1], dtype=Z.dtype, device=Z.device)
    sq = sq.index_add(0, groups, Z * Z)
    norm = sq.clamp_min(1e-20).sqrt()                         # [groups, d]
    scale = (1.0 - thresh[:, None] / norm).clamp_min(0.0)    # [groups, d]
    return Z * scale[groups], scale > 0


def group_lasso_codes(
    x: torch.Tensor,
    B: torch.Tensor,
    groups: torch.Tensor,
    lam: float,
    target_col: torch.Tensor | None = None,
    z0: torch.Tensor | None = None,
    steps: int = 30,
):
    """``(Z [n, d], selected [n, d] bool)`` for rows ``x`` [n, G]; ``selected``
    is the same for all rows of a group.

    ``groups`` [n] are group ids (any integers; relabelled internally).
    ``z0`` is the starting point (e.g. the least-squares activities)."""
    _, g = torch.unique(groups, return_inverse=True)
    n_groups = int(g.max()) + 1 if len(g) else 0
    m = _mask(x, target_col)
    xm = x * m
    counts = torch.bincount(g, minlength=n_groups).to(x.dtype)
    thresh_base = lam * counts.sqrt()
    # step size 1 / L, L = largest eigenvalue of B B^T (masking only lowers it)
    with torch.no_grad():
        L = torch.linalg.eigvalsh(B @ B.T)[-1].clamp_min(1e-12)
    eta = 1.0 / L
    Z = z0 if z0 is not None else torch.zeros(x.shape[0], B.shape[0], dtype=x.dtype, device=x.device)
    Y, t = Z, 1.0
    for _ in range(steps):
        R = (xm - (Y @ B) * m)                   # masked residual
        Zn, _ = _group_prox(Y + eta * (R @ B.T), g, n_groups, eta * thresh_base)
        t_n = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Y = Zn + ((t - 1.0) / t_n) * (Zn - Z)
        Z, t = Zn, t_n
    _, selected = _group_prox(Z, g, n_groups, eta * thresh_base)
    sel_rows = selected[g]                                       # [n, d], same within a group
    return Z, sel_rows


def group_keys(perturbation_idx, context_idx):
    """[n] int64 group ids: one group per (perturbation, context) pair."""
    import numpy as np
    p = np.asarray(perturbation_idx, dtype=np.int64)
    c = np.asarray(context_idx, dtype=np.int64)
    return p * (int(c.max()) + 1 if len(c) else 1) + c


def lasso_groups(perturbation_idx, context_idx, by: str = "pair"):
    """[n] group ids for the lasso. ``"pair"``: one group per (perturbation,
    context) pair -- all rows of a pair share one selection of programs (the
    group lasso). ``"row"``: every row is its own group, so the penalty on a
    program's activity is ``lam * |z_rk|`` (a lasso per row) and a row's
    activities depend on that row alone."""
    import numpy as np
    if by == "pair":
        return group_keys(perturbation_idx, context_idx)
    if by == "row":
        return np.arange(len(perturbation_idx), dtype=np.int64)
    raise ValueError(f"lasso groups must be 'pair' or 'row', got {by!r}")


def group_batches(keys, batch_size, rng=None):
    """Batches (lists of positions into ``keys``) made of whole groups, each of
    about ``batch_size`` rows; groups in random order when ``rng`` is given,
    else sorted."""
    import numpy as np
    keys = np.asarray(keys)
    uniq, inv = np.unique(keys, return_inverse=True)
    if len(uniq) == len(keys):           # one row per group: plain batches
        order = rng.permutation(len(uniq)) if rng is not None else np.arange(len(uniq))
        pos = np.argsort(inv, kind="stable")[order]
        return [pos[i:i + batch_size] for i in range(0, len(pos), batch_size)]
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    order = rng.permutation(len(uniq)) if rng is not None else range(len(uniq))
    batches, cur, n_cur = [], [], 0
    for gi in order:
        cur.append(members[gi]); n_cur += len(members[gi])
        if n_cur >= batch_size:
            batches.append(np.concatenate(cur)); cur, n_cur = [], 0
    if cur:
        batches.append(np.concatenate(cur))
    return batches


def group_penalty(Z: torch.Tensor, groups: torch.Tensor, lam: float) -> torch.Tensor:
    """``lam * sum_g sqrt(n_g) * sum_k ||Z_g[:, k]||_2`` (summed over groups)."""
    _, g = torch.unique(groups, return_inverse=True)
    n_groups = int(g.max()) + 1 if len(g) else 0
    sq = torch.zeros(n_groups, Z.shape[1], dtype=Z.dtype, device=Z.device).index_add(0, g, Z * Z)
    counts = torch.bincount(g, minlength=n_groups).to(Z.dtype)
    return lam * (counts.sqrt()[:, None] * sq.clamp_min(1e-20).sqrt()).sum()
