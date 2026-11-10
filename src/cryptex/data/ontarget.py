"""Which column of ``x`` is a perturbation's own transcript.

Perturbing gene X drops X's own transcript by roughly ``-mean_ctrl[X]``. On one
shard of Stattic, on-target entries have median ``mean_diff`` of ``-0.808``
against ``-0.0001`` for all entries, 96.2% have ``lfsr < 0.05`` against 15.0%,
and 98-99.8% of their magnitude survives shrinkage against 14.2% -- they sit
40-50 standard errors from zero, so the likelihood overwhelms any prior.

These are the strongest and most nearly deterministic entries in the matrix.
Left in, they will claim a factor, and that factor is knockdown efficiency
rather than a regulatory program. Knockdown efficiency is worth modelling
separately -- whether it shifts with drug context is a real question the screen
can answer -- but not as one of the ``d`` programs.
"""

from __future__ import annotations

import numpy as np

from ..models.loadings import NO_MASK


def on_target_index(
    perturbations: np.ndarray,
    genes: np.ndarray,
    control_labels: tuple[str, ...] = ("non-targeting", "NTC", "control"),
) -> np.ndarray:
    """Column index of each row's own gene; :data:`NO_MASK` where there is none.

    Matching is on the bare perturbation name, so it is insensitive to the
    ``__subNN`` suffix that the eleven samples of a perturbation carry -- all
    eleven mask the same column, which is what the spec requires.

    Parameters
    ----------
    perturbations
        [n_rows] perturbation names, with or without a ``__subNN`` suffix.
    genes
        [n_genes] column names of the matrix, in order.
    """
    gene_to_col = {str(g): i for i, g in enumerate(genes)}
    control = {c.lower() for c in control_labels}

    out = np.full(len(perturbations), NO_MASK, dtype=np.int64)
    for i, name in enumerate(perturbations):
        bare = str(name).split("__")[0]
        if bare.lower() in control:
            continue
        col = gene_to_col.get(bare)
        if col is not None:
            out[i] = col
    return out


def mask_coverage(target_col: np.ndarray) -> dict[str, float]:
    """How many rows actually mask something -- worth logging before a fit.

    A low rate means the perturbation names are not matching ``var_names``
    (symbol vs Ensembl ID is the usual cause), in which case the diagonal is
    still in the matrix and will claim a factor.
    """
    masked = int((np.asarray(target_col) != NO_MASK).sum())
    n = len(target_col)
    return {
        "n_rows": float(n),
        "n_masked": float(masked),
        "fraction_masked": masked / n if n else 0.0,
        "n_distinct_genes": float(len(set(np.asarray(target_col).tolist()) - {NO_MASK})),
    }
