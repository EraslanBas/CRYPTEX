"""Held-out evaluation, split at the (perturbation, context) pair level.

This is the test that separates a model which found real structure from one that
memorised a lookup table: predict the LFC vector of a ``(p, c)`` combination the
model never saw. A lookup table scores at chance.

**Split by pair, never by replicate.** Replicates of the same pair share the
cells' underlying state, so a replicate-level split leaks and the score is
meaningless. Same reason bootstrap replicates are discouraged in
``data.pseudoreplicates``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairSplit:
    train: np.ndarray  # boolean mask over rows
    test: np.ndarray
    held_out_pairs: list[tuple[str, str]]


def split_pair_holdout(
    perturbations: np.ndarray,
    contexts: np.ndarray,
    frac: float = 0.1,
    seed: int = 0,
) -> PairSplit:
    """Hold out a random fraction of ``(perturbation, context)`` combinations.

    Every replicate of a held-out pair goes to test, so no pair appears in both
    halves.
    """
    perturbations = np.asarray(perturbations)
    contexts = np.asarray(contexts)
    pairs = np.array(list(zip(perturbations, contexts)), dtype=object)
    unique = sorted({(str(p), str(c)) for p, c in pairs})

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(frac * len(unique))))
    chosen = {unique[i] for i in rng.choice(len(unique), n_test, replace=False)}

    test = np.array([(str(p), str(c)) in chosen for p, c in pairs])
    return PairSplit(train=~test, test=test, held_out_pairs=sorted(chosen))


def split_context_holdout(
    contexts: np.ndarray, held_out: str | list[str]
) -> PairSplit:
    """Hold out entire contexts -- the harder, out-of-domain generalisation test.

    Only meaningful for a model that can extrapolate a context it never saw,
    which requires the context effect to be predictable from something other
    than its own embedding. Expect this to fail for a plain embedding table;
    that failure is informative.
    """
    contexts = np.asarray(contexts).astype(str)
    wanted = {held_out} if isinstance(held_out, str) else set(held_out)
    missing = wanted - set(contexts)
    if missing:
        raise ValueError(f"context(s) not present: {sorted(missing)}")
    test = np.isin(contexts, list(wanted))
    return PairSplit(train=~test, test=test, held_out_pairs=[])


def reconstruction_score(X_true: np.ndarray, X_pred: np.ndarray) -> dict[str, float]:
    """Per-row correlation and variance explained.

    ``mean_row_correlation`` is the headline: how well the model reproduces each
    held-out pair's LFC profile. ``r2`` is computed against the global mean, so
    it is scored against the trivial "no effect" baseline.
    """
    X_true = np.asarray(X_true, dtype=np.float64)
    X_pred = np.asarray(X_pred, dtype=np.float64)
    if X_true.shape != X_pred.shape:
        raise ValueError(f"shape mismatch: {X_true.shape} vs {X_pred.shape}")

    a = X_true - X_true.mean(axis=1, keepdims=True)
    b = X_pred - X_pred.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_corr = np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)

    ss_res = ((X_true - X_pred) ** 2).sum()
    ss_tot = ((X_true - X_true.mean()) ** 2).sum()
    return {
        "mean_row_correlation": float(np.nanmean(row_corr)),
        "median_row_correlation": float(np.nanmedian(row_corr)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
