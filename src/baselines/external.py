"""Adapters for baselines that live in external packages or notebooks.

Each wraps a method that is *not* vendored here, and raises a clear message
naming what to install or extract if it is missing. Kept deliberately thin --
the point is a uniform ``fit``/``transform``/``loadings`` surface, not a
reimplementation.
"""

from __future__ import annotations

import numpy as np

from .registry import register


@register("mofa")
class MOFABaseline:
    """MOFA+ via ``mofapy2``.

    A fitted model and its outputs already exist under ``<root>/results/mofa/``
    (see :mod:`cryptex.paths`)
    (``mofa_model.hdf5``, ``loadings_{A,B,C}.parquet``), produced by
    ``notebooks/baselines/10_MOFA_PerturbSeq.ipynb``. Use
    :meth:`from_hdf5` to wrap those rather than refitting.

    Note MOFA fits per-view loadings, so "global loading matrix" is a choice:
    :meth:`from_hdf5` takes the mean across views by default, which is only
    valid if the views are contexts of the same factor space.
    """

    def __init__(self, n_factors: int = 20, **kwargs):
        self.n_factors = n_factors
        self.kwargs = kwargs

    def fit(self, X, perturbations=None, contexts=None):
        try:
            from mofapy2.run.entry_point import entry_point  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "mofapy2 is required for the MOFA baseline "
                "(`pip install mofapy2`), or wrap an existing fit with "
                "MOFABaseline.from_hdf5(paths.results() / 'mofa/mofa_model.hdf5')"
            ) from exc
        raise NotImplementedError(
            "Refitting MOFA from this adapter is not wired up yet. The fitting "
            "code currently lives in notebooks/baselines/10_MOFA_PerturbSeq.ipynb; "
            "extract it here, or load the existing fit via from_hdf5()."
        )

    @classmethod
    def from_hdf5(cls, path, average_views: bool = True):
        """Wrap an existing MOFA fit written to HDF5."""
        import h5py

        model = cls()
        with h5py.File(path, "r") as f:
            views = list(f["expectations"]["W"])
            W = [np.asarray(f["expectations"]["W"][v]) for v in views]
            Z = np.concatenate(
                [np.asarray(f["expectations"]["Z"][g]) for g in f["expectations"]["Z"]],
                axis=1,
            ).T
        model.loadings_ = np.mean(W, axis=0) if average_views else W
        model.Z_ = Z
        model.views_ = views
        return model

    def transform(self, X):
        if not hasattr(self, "Z_"):
            raise RuntimeError("fit or from_hdf5 first")
        return self.Z_

    @property
    def loadings(self) -> np.ndarray:
        return self.loadings_


@register("muvi")
class MuVIBaseline:
    """MuVI, vendored at ``external/MuVI``.

    Install it in editable mode from there (``pip install -e external/MuVI``)
    rather than copying its source into this tree.
    """

    def __init__(self, n_factors: int = 20, **kwargs):
        self.n_factors = n_factors
        self.kwargs = kwargs

    def fit(self, X, perturbations=None, contexts=None):
        try:
            import muvi  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MuVI is vendored at external/MuVI; install it with "
                "`pip install -e external/MuVI`"
            ) from exc
        raise NotImplementedError(
            "MuVI adapter not wired up yet. MuVI expects prior gene-set masks; "
            "decide what plays that role here (KEGG membership is the obvious "
            "candidate) before implementing."
        )

    def transform(self, X):  # pragma: no cover
        raise NotImplementedError

    @property
    def loadings(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@register("svae_plus")
class SVAEPlusBaseline:
    """sVAE+ (Lopez et al., CLeaR 2023) -- sparse mechanism shift.

    The closest published relative of CRYPTEX: perturbation identity as
    the auxiliary variable, with a sparse perturbation x latent mask. Two
    differences that matter for a fair comparison:

    * it is cell-level with an NB likelihood, so it needs the cell-level h5ads
      rather than the LFC matrices;
    * its decoder is nonlinear (``DecoderSCVI`` with a softmax scale
      activation), so its factor->gene map is *not* global and its loadings are
      local, unlike CRYPTEX's ``B``.

    Interpretable parameters, for reference:
    ``sigmoid(module.action_prior_logit_weight)`` is the [n_perturbations,
    n_latent] targeting probability and ``module.action_prior_mean`` the signed
    shift; the paper binarises the former at 0.5.
    """

    def __init__(self, n_factors: int = 20, **kwargs):
        self.n_factors = n_factors
        self.kwargs = kwargs

    def fit(self, X, perturbations=None, contexts=None):
        try:
            import svae  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sVAE+ is not vendored; clone Genentech/sVAE and install it, or "
                "drop it from the baseline set"
            ) from exc
        raise NotImplementedError(
            "sVAE+ adapter not wired up yet. It consumes cell-level AnnData, "
            "not the (p, c) x gene LFC matrix -- point it at the per-drug h5ads "
            "and note that its sparsity knob (module.sparse_mask_penalty) "
            "defaults to 1, which means NO sparsity."
        )

    def transform(self, X):  # pragma: no cover
        raise NotImplementedError

    @property
    def loadings(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
