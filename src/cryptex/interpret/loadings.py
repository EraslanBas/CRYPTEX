"""Gene-space loading vectors for each factor.

**Never read loadings off the encoder.** Encoder weights (and
``d h_k / d x``) are *filters*, not *patterns*: a filter can put large weight on
a gene precisely to suppress it as a nuisance, so a high encoder weight does not
mean the gene belongs to the factor. This is the standard error documented in
Haufe et al. 2014, "On the interpretation of weight vectors of linear models in
multivariate neuroimaging", and it bites hardest when features are correlated --
which genes are.

Use :func:`loadings_from_decoder` when the model has a global linear decoder
(loadings are then exact model parameters), or :func:`loadings_by_regression`
for any encoder-only model, which recovers the Haufe pattern.

Two indeterminacies survive in every case:

* **Sign** is arbitrary -- flipping ``z_k`` and ``B_k`` together is the same
  model. Use :func:`anchor_signs` before reporting anything directional.
* **Scale** is arbitrary per factor. Relative loadings *within* a factor are
  meaningful; magnitudes *across* factors are not. Rank or z-score within factor
  before comparing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def loadings_from_decoder(B: np.ndarray, genes: np.ndarray) -> pd.DataFrame:
    """Exact loadings from a global linear decoder.

    Already in log-fold-change units: the model is fitted on unscaled shrunken
    logFC, so ``B`` needs no back-transform.

    Parameters
    ----------
    B
        [n_factors, n_genes] decoder weight.
    """
    B = np.asarray(B, dtype=np.float64)
    return pd.DataFrame(
        B,
        index=[f"factor_{k}" for k in range(B.shape[0])],
        columns=np.asarray(genes),
    )


def loadings_by_regression(
    Z: np.ndarray,
    X: np.ndarray,
    genes: np.ndarray,
) -> pd.DataFrame:
    """Haufe pattern: regress gene space on factor activations.

    Row ``k`` is "the average expression profile associated with a unit increase
    in factor ``k``". Use this for encoder-only (discriminative) models, which
    have no decoder to read.
    """
    Z = np.asarray(Z, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    design = np.c_[Z, np.ones(len(Z))]
    coef, *_ = np.linalg.lstsq(design, X, rcond=None)
    return loadings_from_decoder(coef[: Z.shape[1]], genes)


def anchor_signs(
    loadings: pd.DataFrame,
    Z: np.ndarray,
    reference: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Orient each factor and return ``(loadings, signs)``.

    Parameters
    ----------
    reference
        Optional [n_samples] signal of known direction (e.g. activation of a
        perturbation whose effect sign you are confident about). Each factor is
        flipped so its correlation with ``reference`` is positive.

        With no reference, falls back to the convention that the largest-
        magnitude loading is positive. That is *only* a convention -- it makes
        runs comparable, it does not make the direction biologically meaningful.
        For directional claims, supply a reference or report both tails.
    """
    L = loadings.to_numpy().copy()
    if reference is None:
        peak = L[np.arange(len(L)), np.abs(L).argmax(axis=1)]
        signs = np.where(peak < 0, -1.0, 1.0)
    else:
        Z = np.asarray(Z, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        corr = np.array(
            [np.corrcoef(Z[:, k], reference)[0, 1] for k in range(Z.shape[1])]
        )
        corr = np.nan_to_num(corr)
        signs = np.where(corr < 0, -1.0, 1.0)
    return pd.DataFrame(L * signs[:, None], index=loadings.index,
                        columns=loadings.columns), signs


def loadings_per_context(
    Z: np.ndarray,
    X: np.ndarray,
    contexts: np.ndarray,
    genes: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Per-context loadings, fitted independently within each context.

    The model assumes one global map, so these should agree; disagreement is the
    interesting failure. "Factor 7 loads on these genes in DMSO and these others
    under Romidepsin" is a stronger result than one global vector, and it is the
    drug x mechanism interaction a chemogenetic screen is for. Compare against
    the global loadings and check the residual structure per context.
    """
    out = {}
    for context in np.unique(contexts):
        m = contexts == context
        out[str(context)] = loadings_by_regression(Z[m], X[m], genes)
    return out


def loading_agreement(per_context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cosine similarity of each factor's loading vector across contexts.

    Returns a [n_factors, n_contexts] frame of each context's similarity to the
    across-context mean. High agreement (say median > 0.8) justifies reporting a
    single global loading vector; low agreement means one vector per factor is
    misleading and you should report loadings per context.
    """
    names = list(per_context)
    stack = np.stack([per_context[n].to_numpy() for n in names])  # [C, K, G]
    mean = stack.mean(axis=0)
    mean_norm = np.linalg.norm(mean, axis=1)
    sims = np.einsum("ckg,kg->ck", stack, mean) / (
        np.linalg.norm(stack, axis=2) * mean_norm[None, :] + 1e-12
    )
    return pd.DataFrame(
        sims.T,
        index=per_context[names[0]].index,
        columns=names,
    )


def top_genes(
    loadings: pd.DataFrame, factor: str | int, n: int = 50, tail: str = "both"
) -> pd.DataFrame:
    """Highest-|loading| genes for one factor.

    ``tail`` is ``"positive"``, ``"negative"`` or ``"both"``. Because factor sign
    is arbitrary unless anchored, report the two tails separately rather than
    calling one of them "up".
    """
    key = factor if isinstance(factor, str) else f"factor_{factor}"
    row = loadings.loc[key].sort_values()
    if tail == "positive":
        picked = row.tail(n)[::-1]
    elif tail == "negative":
        picked = row.head(n)
    elif tail == "both":
        picked = pd.concat([row.tail(n)[::-1], row.head(n)])
    else:
        raise ValueError("tail must be 'positive', 'negative' or 'both'")
    return picked.rename("loading").to_frame().assign(
        tail=np.where(picked.to_numpy() >= 0, "positive", "negative")
    )
