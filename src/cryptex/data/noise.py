"""Estimation noise from the replicate rows of each pair, and the label-driven subspace.

Every pair ``(perturbation, context)`` has several rows (the full-data
estimate and its re-estimates), and all of them estimate the same response.
They are treated as **exchangeable**: no row is assumed to be the most
precise, and no cell counts are used -- only the matrix and the labels. The
spread of a pair's rows around their mean is estimation noise; pooled over
pairs it gives the within-pair covariance ``Sigma_w``, and a mean of ``k``
rows carries noise ``Sigma_w / k``.

``Sigma_w`` is G x G; it is kept as diagonal + rank-r. The **metric** used
downstream is isotropic + the same rank-r part,

    M = c I + U diag(s) U^T,     c = mean of the per-gene noise variances,

so it discounts noise directions *shared across genes* but leaves every gene's
own scale alone: no per-gene reweighting, in keeping with using the shrunken
logFC exactly as ``ashr`` produced it.

The label-driven subspace solves ``Sigma_S v = lambda M v``: the directions with
the most perturbation-driven variation per unit of correlated noise.
``Sigma_S`` is the second moment of the pair means minus their expected noise.
The eigenvectors ``v`` are filters; the loading subspace is spanned by the
patterns ``a = M v``, returned as the rows of ``V`` in logFC units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseModel:
    """``Sigma_w ~ diag(diag) + U diag(s) U^T``, and the metric built on it."""
    diag: np.ndarray        # [G] per-gene noise variance (of one row)
    U: np.ndarray           # [G, r] orthonormal correlated-noise directions
    s: np.ndarray           # [r] their variances (of one row)

    @property
    def c(self) -> float:
        """Isotropic level of the metric: the mean per-gene noise variance."""
        return float(self.diag.mean())

    def metric_inv_quadratic(self, R: np.ndarray) -> np.ndarray:
        """Per-row ``r M^{-1} r^T`` for rows of ``R``, by Woodbury."""
        c = self.c
        proj = R @ self.U
        return ((R * R).sum(1) - (proj * proj * (self.s / (c + self.s))).sum(1)) / c

    def metric_inv_sqrt(self, X: np.ndarray) -> np.ndarray:
        """``X M^{-1/2}``."""
        c = self.c
        k = 1.0 - np.sqrt(c / (c + self.s))
        return (X - (X @ self.U) * k @ self.U.T) / np.sqrt(c)

    def metric_sqrt(self, X: np.ndarray) -> np.ndarray:
        """``X M^{1/2}``."""
        c = self.c
        k = np.sqrt((c + self.s) / c) - 1.0
        return (X + (X @ self.U) * k @ self.U.T) * np.sqrt(c)

    def cell_cov_apply(self, X: np.ndarray) -> np.ndarray:
        """``X Sigma_w``."""
        return X * self.diag + (X @ self.U) * self.s @ self.U.T


def _rows_index(rows, is_real, n):
    rows = np.nonzero(rows)[0] if np.asarray(rows).dtype == bool else np.asarray(rows, dtype=np.int64)
    if is_real is not None:
        rows = rows[np.asarray(is_real, dtype=bool)[rows]]
    return rows


def pair_means(X: np.ndarray, pair: np.ndarray, rows: np.ndarray,
               is_real: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(keys, means [pairs, G], counts [pairs])`` over the measured rows among
    ``rows``. Every row of a pair counts equally."""
    rows = _rows_index(rows, is_real, len(X))
    keys, inv, counts = np.unique(np.asarray(pair)[rows], return_inverse=True, return_counts=True)
    sums = np.zeros((len(keys), X.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, np.asarray(X[rows], dtype=np.float64))
    return keys, sums / counts[:, None], counts


def within_pair_residuals(X: np.ndarray, pair: np.ndarray, rows: np.ndarray,
                          max_rows: int | None = None, seed: int = 0,
                          is_real: np.ndarray | None = None) -> np.ndarray:
    """Each measured row minus its pair's mean, over pairs with at least two
    rows, scaled by ``sqrt(k / (k - 1))`` so that every returned row has the
    within-pair covariance ``Sigma_w`` (a deviation from a mean of ``k``
    exchangeable rows has covariance ``(k - 1) / k * Sigma_w``)."""
    rows = _rows_index(rows, is_real, len(X))
    keys, means, counts = pair_means(X, pair, rows)
    inv = np.searchsorted(keys, np.asarray(pair)[rows])
    keep = counts[inv] >= 2
    rows, inv = rows[keep], inv[keep]
    if not len(rows):
        raise ValueError("no pair has two or more rows; the noise model needs replicates")
    if max_rows is not None and len(rows) > max_rows:
        pick = np.sort(np.random.default_rng(seed).choice(len(rows), max_rows, replace=False))
        rows, inv = rows[pick], inv[pick]
    k = counts[inv].astype(np.float64)
    R = (np.asarray(X[rows], dtype=np.float64) - means[inv]) * np.sqrt(k / (k - 1))[:, None]
    return R.astype(np.float32)


def noise_covariance(R: np.ndarray, rank: int = 50, seed: int = 0) -> NoiseModel:
    """Diagonal + rank-``rank`` fit of the covariance of the residuals ``R``.

    The residuals have mean zero by construction, so second moments are used.
    The low-rank part is the top principal directions of ``R``; the diagonal is
    what those leave of each gene's variance (floored at a small fraction of the
    mean, so no gene's noise is zero).
    """
    import torch

    Rt = torch.from_numpy(np.asarray(R, dtype=np.float32))
    n = Rt.shape[0]
    total = (Rt * Rt).sum(0).double().numpy() / n
    torch.manual_seed(seed)
    _, S, V = torch.svd_lowrank(Rt, q=rank + 10, niter=4)
    U = V[:, :rank].double().numpy()
    lam = (S[:rank].double().numpy() ** 2) / n     # variance along each u_k
    # Each leading direction's variance also contains the per-gene noise along
    # it, u_k^T diag u_k; separate the two by a few factor-analysis-style
    # alternations (otherwise s is biased up by about the mean gene variance).
    floor = 1e-3 * total.mean()
    diag = np.maximum(total - (U * U * lam).sum(1), floor)
    for _ in range(5):
        s = np.maximum(lam - (U * U * diag[:, None]).sum(0), 0.0)
        diag = np.maximum(total - (U * U * s).sum(1), floor)
    return NoiseModel(diag=diag, U=U, s=s)


def label_subspace(
    means: np.ndarray, counts: np.ndarray, noise: NoiseModel, d: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(V, eigenvalues)``: the top-``d`` label-driven loading directions.

    ``means`` [pairs, G] are the training pairs' mean rows and ``counts`` how
    many rows each mean averages. A mean of ``k`` exchangeable rows carries
    noise ``Sigma_w / k``, so ``Sigma_S = E[m^T m] - Sigma_w E[1/k]``. The
    generalised eigenproblem ``Sigma_S v = lambda M v`` is solved in the
    ``M``-whitened coordinates, and the patterns ``a = M v`` are returned as the
    rows of ``V`` [d, G], each scaled to unit norm.
    """
    Xw = noise.metric_inv_sqrt(np.asarray(means, dtype=np.float64))
    S_w = Xw.T @ Xw / len(Xw)
    # expected noise in whitened coordinates: M^-1/2 Sigma_w M^-1/2 * E[1/k]
    I = np.eye(Xw.shape[1])
    Nw = noise.metric_inv_sqrt(noise.cell_cov_apply(noise.metric_inv_sqrt(I)))
    S_w -= Nw * float(np.mean(1.0 / np.asarray(counts, dtype=float)))
    S_w = 0.5 * (S_w + S_w.T)
    evals, evecs = np.linalg.eigh(S_w)
    order = np.argsort(evals)[::-1][:d]
    W = evecs[:, order].T                      # [d, G], whitened eigenvectors
    V = noise.metric_sqrt(W)                   # patterns a = M v = M^{1/2} w
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V.astype(np.float32), evals[order]


def label_subspace_from_rows(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    rows: np.ndarray, d: int, rank: int = 50, max_rows: int | None = 150_000,
    seed: int = 0, is_real: np.ndarray | None = None,
) -> tuple[np.ndarray, NoiseModel]:
    """Noise model and label-driven subspace ``V`` [d, G] from the measured
    rows among ``rows`` (the training rows): within-pair residuals for the
    noise, pair means for the signal. Uses only the matrix and the
    ``(perturbation, context)`` labels -- no cell counts, and no row of a pair
    is treated as better than another. What :func:`cryptex.train.fit` uses when
    ``subspace="fixed"`` and no basis is given."""
    pair = np.asarray(perturbation_idx, dtype=np.int64) * (int(np.max(context_idx)) + 1) \
        + np.asarray(context_idx, dtype=np.int64)
    R = within_pair_residuals(X, pair, rows, max_rows, seed, is_real)
    noise = noise_covariance(R, rank=min(rank, R.shape[0] - 1, R.shape[1] - 1), seed=seed)
    del R
    _, means, counts = pair_means(X, pair, rows, is_real)
    V, _ = label_subspace(means, counts, noise, d)
    return V, noise


def rca_subspace(
    X: np.ndarray, pair: np.ndarray, rows: np.ndarray, d: int, rank: int = 100,
    seed: int = 0, is_real: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(V [d, G], reliabilities [d])`` by reliable components analysis.

    Dmochowski et al. (2012, 2015), with a (perturbation, context) pair as the
    stimulus and its rows as the repeats. Over the measured rows among
    ``rows``:

        R_B = sum_pairs sum_{r != s} x_r^T x_s      (across repeats: shared part)
        R_W = sum_pairs (k - 1) sum_r x_r^T x_r      (within repeats: signal + noise)

    The ``(k - 1)`` weight is the number of row pairs each row belongs to, so
    both sums run over the same pairs of rows; pairs with a single row form no
    pair and drop out. The components maximise the reliability
    ``rho = w R_B w^T / w R_W w^T`` (in [-1, 1]), i.e. solve
    ``R_B w = rho R_W w``. As in RCA, ``R_W`` is regularised by truncation to its
    top ``rank`` eigen-directions: the problem is solved in that subspace. The
    rows of ``V`` are the forward-model patterns ``w R_W``, each of unit norm.
    No centring (zero is "no response" for logFC) and no cell counts.
    """
    rows = _rows_index(rows, is_real, len(X))
    keys, inv, counts = np.unique(np.asarray(pair)[rows], return_inverse=True, return_counts=True)
    keep = counts[inv] >= 2
    rows, inv = rows[keep], inv[keep]
    if not len(rows):
        raise ValueError("no pair has two or more rows; RCA needs repeats")
    k = counts[inv].astype(np.float64)
    Xr = np.asarray(X[rows], dtype=np.float64)
    # top `rank` eigen-directions of R_W = Xw^T Xw, Xw rows scaled by sqrt(k - 1)
    from sklearn.utils.extmath import randomized_svd
    Xw = Xr * np.sqrt(k - 1)[:, None]
    _, sw, Ut = randomized_svd(Xw.astype(np.float32), min(rank, *Xw.shape) , n_iter=6, random_state=seed)
    U = Ut.T.astype(np.float64)                          # [G, K]
    Rw = np.diag(sw.astype(np.float64) ** 2)             # R_W in that basis
    # R_B = sum_pairs (k^2 xbar^T xbar - sum_r x_r^T x_r), in the same basis
    Y = Xr @ U                                           # rows in the K-dim basis
    sums = np.zeros((len(keys), U.shape[1])); np.add.at(sums, inv, Y)
    Rb = sums.T @ sums - Y.T @ Y
    Rb = 0.5 * (Rb + Rb.T)
    # R_B w = rho R_W w with R_W diagonal: whiten, then a symmetric eigenproblem
    iw = 1.0 / np.sqrt(np.diag(Rw))
    rho, E = np.linalg.eigh(Rb * iw[:, None] * iw[None, :])
    order = np.argsort(rho)[::-1][:d]
    Wk = E[:, order] * iw[:, None]                       # filters in the K basis
    V = (U @ (Rw @ Wk)).T                                # patterns w R_W, back in genes
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V.astype(np.float32), rho[order]


def rca_subspace_from_rows(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    rows: np.ndarray, d: int, rank: int = 100, seed: int = 0,
    is_real: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`rca_subspace` with the pair key built from the label indices."""
    pair = np.asarray(perturbation_idx, dtype=np.int64) * (int(np.max(context_idx)) + 1) \
        + np.asarray(context_idx, dtype=np.int64)
    return rca_subspace(X, pair, rows, d, rank=rank, seed=seed, is_real=is_real)[0]


def pca_subspace(
    X: np.ndarray, pair: np.ndarray, rows: np.ndarray, d: int,
    is_real: np.ndarray | None = None, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(V [d, G], singular values [d])``: PCA of the pair means.

    Each pair's measured rows (among ``rows``) are averaged, every row
    counting the same; ``V`` holds the top ``d`` right singular vectors of the
    pair-mean matrix (unit-norm rows). Not centred: zero is "no response" for
    logFC, and centring would centre every gene. No noise model and no cell
    counts. On the real screen this subspace is markedly more stable across
    halves of the pairs than the signal-to-noise estimates.
    """
    from sklearn.utils.extmath import randomized_svd
    _, means, _ = pair_means(X, pair, rows, is_real)
    _, s, Vt = randomized_svd(means.astype(np.float32), d, n_iter=7, random_state=seed)
    V = Vt / np.linalg.norm(Vt, axis=1, keepdims=True)
    return V.astype(np.float32), s


def pca_subspace_from_rows(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    rows: np.ndarray, d: int, seed: int = 0, is_real: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`pca_subspace` with the pair key built from the label indices."""
    pair = np.asarray(perturbation_idx, dtype=np.int64) * (int(np.max(context_idx)) + 1) \
        + np.asarray(context_idx, dtype=np.int64)
    return pca_subspace(X, pair, rows, d, is_real=is_real, seed=seed)[0]
