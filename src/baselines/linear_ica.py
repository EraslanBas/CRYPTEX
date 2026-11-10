"""Linear ICA baselines -- the ones CRYPTEX has to beat.

With a global linear decoder and a linear encoder, CRYPTEX *is* linear ICA
with auxiliary variables. So these are not weak strawmen: they estimate the same
model class by cheaper means, and they run in minutes on the rowbound matrix.

The contrastive model earns its keep only if it exploits information these
cannot: distribution shape beyond second moments (via the non-quadratic basis
statistics), and the perturbation x context interaction through the label net.
Run these first.

Two identifiability routes, matching the two available assumptions:

* :class:`FastICABaseline` -- non-Gaussianity of the factors.
* :class:`ContextJointDiagonalization` -- context-varying second-order
  structure, i.e. the nonstationarity route, which uses the 16-context design as
  the identifying structure rather than treating context as a nuisance.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, FastICA

from .joint_diagonalization import joint_diagonalize, off_diagonal_energy
from .registry import register


@register("linear_ica")
class FastICABaseline:
    """PCA whitening followed by FastICA.

    Parameters
    ----------
    n_factors
        Number of components.
    n_whiten
        PCA dimension before ICA. Standard practice, and cheap here; defaults to
        ``max(n_factors, 50)``.
    """

    def __init__(
        self,
        n_factors: int = 20,
        n_whiten: int | None = None,
        max_iter: int = 1000,
        random_state: int = 0,
    ):
        self.n_factors = n_factors
        self.n_whiten = n_whiten if n_whiten is not None else max(n_factors, 50)
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, perturbations=None, contexts=None):
        X = np.asarray(X, dtype=np.float64)
        n_whiten = min(self.n_whiten, *X.shape)
        self.pca_ = PCA(n_components=n_whiten, random_state=self.random_state).fit(X)
        scores = self.pca_.transform(X)
        self.ica_ = FastICA(
            n_components=self.n_factors,
            max_iter=self.max_iter,
            random_state=self.random_state,
            whiten="unit-variance",
        ).fit(scores)
        # factor -> gene loadings, mapped back through the PCA basis
        self.loadings_ = self.ica_.mixing_.T @ self.pca_.components_
        return self

    def transform(self, X):
        return self.ica_.transform(self.pca_.transform(np.asarray(X, dtype=np.float64)))

    @property
    def loadings(self) -> np.ndarray:
        return self.loadings_


@register("context_jd")
class ContextJointDiagonalization:
    """Joint diagonalization of per-context covariance matrices.

    Identifies the mixing from *nonstationarity across contexts* rather than
    from non-Gaussianity: if each context shifts the factor variances, the
    per-context covariances share one diagonalizing basis, and that basis is the
    unmixing. Needs no distributional assumption beyond second moments.

    Check :attr:`residual_off_diagonal_` after fitting. It is the fraction of
    energy left off-diagonal, so a value near 0 means the covariances genuinely
    share a basis and the result is trustworthy; a large value means they do not
    and the factors are not identified by this route.
    """

    def __init__(self, n_factors: int = 20, random_state: int = 0):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, X, perturbations=None, contexts=None):
        if contexts is None:
            raise ValueError("context_jd requires `contexts`")
        X = np.asarray(X, dtype=np.float64)
        contexts = np.asarray(contexts)
        unique = np.unique(contexts)
        if len(unique) < 2:
            raise ValueError(
                f"need >= 2 contexts to identify anything, got {len(unique)}"
            )

        self.pca_ = PCA(
            n_components=min(self.n_factors, *X.shape), whiten=True,
            random_state=self.random_state,
        ).fit(X)
        W = self.pca_.transform(X)

        covs = np.stack([np.cov(W[contexts == c], rowvar=False) for c in unique])
        V, _ = joint_diagonalize(covs)
        self.residual_off_diagonal_ = off_diagonal_energy(covs, V)
        self.contexts_ = unique
        self.unmixing_ = V.T
        self.loadings_ = V.T @ (
            self.pca_.components_ * np.sqrt(self.pca_.explained_variance_)[:, None]
        )
        return self

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, dtype=np.float64)) @ self.unmixing_.T

    @property
    def loadings(self) -> np.ndarray:
        return self.loadings_


@register("pca")
class PCABaseline:
    """Plain PCA -- the "no identifiability at all" reference point.

    Included because it is the honest floor: PCA reduces dimension perfectly
    well but its individual axes carry no meaning beyond the subspace they span.
    If a method cannot beat PCA on held-out (p, c) reconstruction, its factors
    are not worth interpreting.
    """

    def __init__(self, n_factors: int = 20, random_state: int = 0):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, X, perturbations=None, contexts=None):
        X = np.asarray(X, dtype=np.float64)
        self.pca_ = PCA(
            n_components=min(self.n_factors, *X.shape), random_state=self.random_state
        ).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, dtype=np.float64))

    @property
    def loadings(self) -> np.ndarray:
        return self.pca_.components_
