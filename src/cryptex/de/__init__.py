"""The ChemoGeneticScreens shrunken-logFC pipeline, vendored unchanged.

These are the scripts that generated the published ``PosteriorMeanMatrices/``.
They are standalone CLIs, not importable helpers -- driven by
``RunAshrPipeline.sh``, which chains all four steps per input ``.h5ad``:

1. ``ComputeSE.py``          cell-level -> per-perturbation ``mean_diff`` + Welch
                             ``se``, written as ``chunk_%06d.csv``
                             (byte-identical to
                             ``ChemoGeneticScreens/SRC/PythonScripts/ComputeSE.py``)
2. ``run_ashr_on_chunk.R``   ``ash(a$mean_diff, a$se)`` per chunk, so the
                             empirical-Bayes prior is fitted per 100
                             perturbations
3. ``MergeAshrRes.py``       merge perturbation/gene metadata back into the ashr
                             output
4. ``AshrGenerateMatrix.py`` pivot to the wide perturbation x gene
                             ``PosteriorMean`` matrix

Usage::

    OUT_BASE=data/posterior_matrices \\
      src/cryptex/de/RunAshrPipeline.sh /path/to/one.h5ad /path/to/two.h5ad

**Requires R with ``ashr``.** It is not installed in this environment; step 2
fails without ``R -e 'install.packages("ashr")'``.

Two Python modules here are not part of the original pipeline, and neither does
any differential expression:

* ``arms.py``  builds the *augmented input* -- one label per intended matrix row
  (``<pert>`` for the full estimate, ``<pert>__sub<k>`` for subsamples), so full
  and subsampled results land in the same matrix and the same ashr fit.
* ``api.py``   one call that chains input construction -> ``RunAshrPipeline.sh``
  -> matrix + per-row cell counts.
"""

from .api import (
    PosteriorRun,
    context_path,
    generate_posterior_matrices,
    run_screen,
)
from .matrices import (
    build_all_matrices,
    build_context,
    load_matrices,
    load_shard_matrix,
)
from .shards import (
    ShardResult,
    compute_se_for_shard,
    context_seed,
    find_shards,
    plan_shard_rows,
    prepare_control_subset,
)
from .arms import (
    MIN_CELLS_TO_SUBSAMPLE,
    N_CONTROL_CELLS,
    SUBSAMPLE_SEP,
    build_augmented_adata,
    plan_augmentation,
    subsample_sizes,
)

__all__ = [
    "generate_posterior_matrices",
    "run_screen",
    "context_path",
    "PosteriorRun",
    "build_augmented_adata",
    "plan_augmentation",
    "subsample_sizes",
    "SUBSAMPLE_SEP",
    "MIN_CELLS_TO_SUBSAMPLE",
    "find_shards",
    "prepare_control_subset",
    "plan_shard_rows",
    "compute_se_for_shard",
    "ShardResult",
    "build_all_matrices",
    "load_matrices",
    "build_context",
    "load_shard_matrix",
    "context_seed",
    "N_CONTROL_CELLS",
]
