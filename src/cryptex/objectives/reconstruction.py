"""On-target-masked reconstruction error, averaged over rows.

The per-row error (``GlobalLoadings.squared_error``) is the masked squared
residual divided by the number of retained genes; this averages it over a
batch's rows. Every row weighs the same: all rows of a pair are exchangeable,
and the model never uses cell counts.
"""

from __future__ import annotations

import torch


def weighted_squared_error(
    per_row_error: torch.Tensor, row_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Weighted mean of per-row errors.

    ``per_row_error`` comes from
    :meth:`cryptex.models.loadings.GlobalLoadings.squared_error`, which
    has already applied the on-target mask and divided by the retained gene
    count. Weights are renormalised within the batch so ``alpha`` means the
    same thing at any batch size.
    """
    if row_weights is None:
        return per_row_error.mean()
    w = row_weights / row_weights.sum().clamp_min(1e-12)
    return (per_row_error * w).sum()
