"""Canonical locations for CRYPTEX data.

The repository holds *code only*. Every data artifact -- screen intermediates,
posterior-mean matrices, augmented tables, simulator output -- lives under a
single root on large_storage.

Override the root with the ``CRYPTEX_ROOT`` environment variable::

    export CRYPTEX_ROOT=/some/other/place

Layout::

    <root>/
      computese/   ComputeSE + ashr intermediates, one dir per context  (regenerable)
      matrices/    <context>_PosteriorMean.parquet + <context>_row_metadata.csv
      augmented/   permuted-label tables and column-shuffled matrices
      simulator/   perturb-seq simulator artifacts (h5ad, grn.pkl, ground truth)
      results/     baseline model fits (mofa/, de_chunks/)
      logs/        driver and builder logs
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = "/large_storage/ctc/beraslan/ModuleFinder"


def root() -> Path:
    """Data root; ``CRYPTEX_ROOT`` (or the older ``EXTRACT_ROOT`` / ``MODULEFINDER_ROOT``)
    wins over :data:`DEFAULT_ROOT`."""
    return Path(os.environ.get("CRYPTEX_ROOT")
                or os.environ.get("EXTRACT_ROOT")
                or os.environ.get("MODULEFINDER_ROOT")
                or DEFAULT_ROOT)


def _sub(name: str, create: bool = False) -> Path:
    p = root() / name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def computese(create: bool = False) -> Path:
    return _sub("computese", create)


def matrices(create: bool = False) -> Path:
    return _sub("matrices", create)


def augmented(create: bool = False) -> Path:
    return _sub("augmented", create)


def simulator(create: bool = False) -> Path:
    return _sub("simulator", create)


def results(create: bool = False) -> Path:
    return _sub("results", create)


def logs(create: bool = False) -> Path:
    return _sub("logs", create)
