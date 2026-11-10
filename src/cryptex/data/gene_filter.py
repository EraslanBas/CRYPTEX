"""Drop genes no perturbation moved, in any context.

A gene that no perturbation in the panel shifted anywhere is not part of any
latent factor that was active in any context, so it contributes nothing but
noise to the factorisation. Counting how many perturbations moved each gene, and
requiring a minimum, removes those.

The count is over ``(perturbation, context)`` pairs, summed across contexts, and
it uses each pair's **full-data row only**. The eleven subsample rows of a pair
are re-estimates of the same quantity, so counting them would multiply every
gene's count by roughly eleven without adding information, and would count a
gene as "affected in many perturbations" when one poorly powered perturbation
happened to cross the threshold in several of its own subsamples.

"Affected" has to be a magnitude threshold, because the built matrices carry
``PosteriorMean`` and not ``lfsr``. That threshold is part of the definition, not
an incidental default: ``ashr`` shrinks unsupported entries to near zero, so the
count is stable for thresholds well above the shrinkage floor and grows quickly
below it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def affected_counts(
    X: pd.DataFrame | np.ndarray,
    meta: pd.DataFrame,
    threshold: float = 0.1,
    variant: str = "main",
) -> pd.Series:
    """Number of ``(perturbation, context)`` pairs that moved each gene.

    Parameters
    ----------
    X
        ``[rows x genes]``, as returned by
        :func:`cryptex.de.load_matrices`.
    meta
        Row metadata with ``variant``; must align row-for-row with ``X``.
    threshold
        A gene counts as affected in a pair when ``|value| > threshold``.
    variant
        Which rows to count. ``"main"`` uses one full-data row per pair, which
        is the intended behaviour. Pass ``None`` to count every row.

    Returns
    -------
    Counts indexed by gene, in the column order of ``X``.
    """
    if len(meta) != (X.shape[0] if hasattr(X, "shape") else len(X)):
        raise ValueError("meta and X must align row-for-row")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    genes = (
        np.asarray(X.columns)
        if isinstance(X, pd.DataFrame)
        else np.arange(np.asarray(X).shape[1])
    )
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

    if variant is not None:
        if "variant" not in meta:
            raise KeyError("meta has no 'variant' column")
        keep = (meta["variant"].to_numpy() == variant)
        if not keep.any():
            raise ValueError(f"no rows with variant == {variant!r}")
        values = values[keep]

    counts = np.zeros(values.shape[1], dtype=np.int64)
    for start in range(0, len(values), 4096):      # chunked: X is ~12 GB whole
        block = values[start : start + 4096]
        counts += (np.abs(block) > threshold).sum(axis=0)
    return pd.Series(counts, index=genes, name="n_affected")


def select_genes(
    X: pd.DataFrame | np.ndarray,
    meta: pd.DataFrame,
    min_perturbations: int,
    threshold: float = 0.1,
    variant: str = "main",
) -> np.ndarray:
    """Boolean mask over genes: ``True`` where at least
    ``min_perturbations`` pairs moved the gene by more than ``threshold``."""
    if min_perturbations < 1:
        raise ValueError("min_perturbations must be at least 1")
    counts = affected_counts(X, meta, threshold=threshold, variant=variant)
    return (counts.to_numpy() >= min_perturbations)


def filter_summary(
    counts: pd.Series, thresholds: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)
) -> pd.DataFrame:
    """How many genes survive each choice of ``min_perturbations``.

    Report this before fixing ``N``: the right value depends on where the count
    distribution falls away, which is a property of the screen.
    """
    n = len(counts)
    rows = [
        {
            "min_perturbations": t,
            "genes_kept": int((counts >= t).sum()),
            "fraction_kept": float((counts >= t).mean()),
            "genes_dropped": int(n - (counts >= t).sum()),
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def compute_gene_list(
    matrices_dir,
    min_affected: int,
    threshold: float = 0.1,
    contexts: list[str] | None = None,
    out_path=None,
) -> pd.DataFrame:
    """Decide which genes to keep, from the **real** matrices, and save the list.

    The decision is made on real data only. Deriving it from the augmented
    tables instead would be circular: the column-shuffled matrices have the
    same per-gene marginals by construction, so they carry exactly the same
    counts and add nothing, while the permuted-label tables hold no expression
    values at all.

    Saving matters because the counts depend on which contexts exist when they
    are computed. A context added later can only raise a gene's count, so the
    surviving set grows, and two fits run either side of that change would use
    different gene sets with no way to notice. Fix the list once, version it,
    and have every consumer read it.

    Returns a frame of ``gene, n_affected, keep`` for every gene, and writes it
    to ``out_path`` as TSV when given.
    """
    from pathlib import Path

    from ..de import load_matrices

    X, meta = load_matrices(matrices_dir, contexts=contexts, variant="main")
    counts = affected_counts(X, meta, threshold=threshold, variant="main")
    return gene_list_table(counts, meta, min_affected, threshold, out_path)


def gene_list_table(
    counts: pd.Series,
    meta: pd.DataFrame,
    min_affected: int,
    threshold: float,
    out_path=None,
) -> pd.DataFrame:
    """``gene, n_affected, keep`` from counts; written as TSV when ``out_path``
    is given, in the format :func:`load_gene_list` reads.

    ``meta`` is the full-data rows the counts came from; it is recorded in the
    header so the list says which pairs decided it.
    """
    from pathlib import Path

    out = pd.DataFrame(
        {
            "gene": counts.index.to_numpy(),
            "n_affected": counts.to_numpy(),
            "keep": counts.to_numpy() >= min_affected,
        }
    )
    out.attrs.update(
        min_affected=min_affected,
        threshold=threshold,
        n_pairs=int(len(meta)),
        contexts=sorted(meta.context.unique().tolist()),
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# min_affected={min_affected}\tthreshold={threshold}\n"
            f"# n_pairs={len(meta)}\tcontexts={','.join(sorted(meta.context.unique()))}\n"
            f"# kept={int(out.keep.sum())}/{len(out)}\n"
        )
        with open(out_path, "w") as fh:
            fh.write(header)
            out.to_csv(fh, sep="\t", index=False)
    return out


def load_gene_list(path) -> np.ndarray:
    """Read the kept genes from a file written by :func:`compute_gene_list`."""
    table = pd.read_csv(path, sep="\t", comment="#")
    if "keep" not in table or "gene" not in table:
        raise ValueError(f"{path} is not a gene list written by compute_gene_list")
    return table.loc[table["keep"], "gene"].to_numpy()


def apply_gene_list(
    X: pd.DataFrame | np.ndarray,
    gene_list: np.ndarray,
    genes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subset columns to ``gene_list``, in the list's order.

    Use this for the real matrices and for the augmented ones, so both carry
    the same columns in the same order. Ordering by the list rather than by the
    input means two files with different column orders still line up, and a
    missing gene is an error rather than a silent misalignment.
    """
    if isinstance(X, pd.DataFrame):
        genes = np.asarray(X.columns)
        values = X.to_numpy()
    else:
        if genes is None:
            raise ValueError("pass `genes` when X is not a DataFrame")
        genes = np.asarray(genes)
        values = np.asarray(X)
    if values.shape[1] != len(genes):
        raise ValueError(
            f"X has {values.shape[1]} columns but {len(genes)} gene names"
        )

    position = {g: i for i, g in enumerate(genes)}
    missing = [g for g in gene_list if g not in position]
    if missing:
        raise KeyError(
            f"{len(missing)} genes from the list are absent, e.g. {missing[:5]}"
        )
    idx = np.array([position[g] for g in gene_list], dtype=np.int64)
    return values[:, idx], np.asarray(gene_list)
