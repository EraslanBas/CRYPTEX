"""Shard-wise ComputeSE, using the SPLIT layout the pipeline was designed for.

``SplitBigAnndata.py`` partitions each drug by perturbation -- 100 per shard,
shards holding **only** perturbed cells -- and writes the controls once to
``<ctx>_controls.h5ad``. Both are dense float32. That decomposition is why the
original runs were cheap: a shard is ~7.7 GB and the controls ~24 GB, against
~241 GB for the whole drug (and ~960 GB once augmented, which does not fit).

Per shard here: load shard + the persisted control subset, duplicate the
subsampled cells, and run ``ComputeSE.py`` on the combination. Peak ~43 GB, so
many shards can run at once.

``ComputeSE.py`` always names its output ``chunk_%06d.csv`` starting at 0, so
each shard writes into its own subdirectory to avoid clobbering.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .arms import MIN_CELLS_TO_SUBSAMPLE, SUBSAMPLE_SEP, disjoint_replicates, subsample_sizes

COMPUTE_SE = Path(__file__).resolve().parent / "ComputeSE.py"
LABEL_KEY = "mf_label"


def context_seed(context: str, seed: int = 0) -> int:
    """Stable per-context seed. ``hash()`` is salted per process, so don't."""
    digest = hashlib.md5(f"{context}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def find_shards(screen_dir: str | Path, context: str) -> tuple[list[Path], Path]:
    """``(shard_paths, controls_path)`` from ``<ctx>/SPLIT/``."""
    split = Path(screen_dir) / context / "SPLIT"
    if not split.is_dir():
        raise FileNotFoundError(f"no SPLIT directory: {split}")
    shards = sorted(split.glob("ad_*.h5ad"))
    if not shards:
        raise FileNotFoundError(f"no ad_*.h5ad shards under {split}")
    controls = split / f"{context}_controls.h5ad"
    if not controls.exists():
        candidates = sorted(split.glob("*_controls.h5ad"))
        if not candidates:
            raise FileNotFoundError(f"no controls file under {split}")
        controls = candidates[0]
    return shards, controls


def prepare_control_subset(
    screen_dir: str | Path,
    context: str,
    out_dir: str | Path,
    n_control_cells: int = 100_000,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Sample ``n_control_cells`` controls **once** per context and persist them.

    Written to ``<out_dir>/controls/<context>_controls_<n>.h5ad`` and reused on
    every later call, so every shard of a context -- and every re-run -- is
    measured against exactly the same baseline. Without this, rows from
    different shards would not be comparable.
    """
    import anndata as ad

    out_path = Path(out_dir) / "controls" / f"{context}_controls_{n_control_cells}.h5ad"
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _, controls_path = find_shards(screen_dir, context)
    backed = ad.read_h5ad(controls_path, backed="r")
    available = backed.n_obs
    if available <= n_control_cells:
        chosen = np.arange(available)
        if available < n_control_cells:
            print(
                f"[warn] {context}: only {available:,} control cells available, "
                f"fewer than the requested {n_control_cells:,}; using all.",
                flush=True,
            )
    else:
        rng = np.random.default_rng(context_seed(context, seed))
        chosen = np.sort(rng.choice(available, n_control_cells, replace=False))

    subset = backed[chosen].to_memory()
    backed.file.close()
    # Write then rename: a run killed mid-write would otherwise leave a
    # truncated file that the exists() check above would silently reuse,
    # and every shard of the context would be measured against garbage.
    tmp_path = out_path.with_suffix(".h5ad.tmp")
    subset.write_h5ad(tmp_path)
    tmp_path.replace(out_path)
    print(
        f"[controls] {context}: {len(chosen):,} of {available:,} -> {out_path.name} "
        f"({out_path.stat().st_size/1e9:.1f} GB)",
        flush=True,
    )
    return out_path


def plan_shard_rows(
    perturbations: np.ndarray,
    n_subsamples: int,
    min_cells_to_subsample: int = MIN_CELLS_TO_SUBSAMPLE,
    spacing: str = "linear",
    seed: int = 0,
    scheme: str = "nested",
    replicate_cells: int = 100,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Plan the rows for one shard. Shards carry no controls, so none are handled.

    ``scheme="nested"`` (the original matrices): ``n_subsamples`` random
    subsets with sizes spanning ``[min_cells_to_subsample, n)``, drawn
    independently, so they overlap each other. ``scheme="disjoint"``: up to
    ``n_subsamples`` non-overlapping pseudobulks of exactly ``replicate_cells``
    cells each (:func:`~cryptex.de.arms.disjoint_replicates`). Both keep the
    full row and label the extra rows ``<pert>__sub<k>``, so everything
    downstream reads them the same way; ``row_plan.csv`` records the scheme.

    Returns ``(cell_index, labels, plan)`` over the shard's own cells, with
    subsampled cells repeated.
    """
    if scheme not in ("nested", "disjoint"):
        raise ValueError(f"scheme must be 'nested' or 'disjoint', got {scheme!r}")
    perturbations = np.asarray(perturbations).astype(str)
    rng = np.random.default_rng(seed)
    cell_index = [np.arange(len(perturbations))]
    labels = [perturbations.copy()]
    rows = []

    for pert in np.unique(perturbations):
        idx = np.flatnonzero(perturbations == pert)
        rows.append({"label": pert, "perturbation": pert, "variant": "full",
                     "subsample": -1, "n_cells_planned": len(idx), "scheme": scheme})
        if scheme == "disjoint":
            groups = disjoint_replicates(idx, replicate_cells, n_subsamples, rng)
        else:
            groups = [rng.choice(idx, size=n, replace=False) for n in
                      subsample_sizes(len(idx), n_subsamples, min_cells_to_subsample, spacing)]
        for k, chosen in enumerate(groups):
            n = len(chosen)
            label = f"{pert}{SUBSAMPLE_SEP}{k:02d}"
            cell_index.append(chosen)
            labels.append(np.full(n, label))
            rows.append({"label": label, "perturbation": pert, "variant": "subsample",
                         "subsample": k, "n_cells_planned": n, "scheme": scheme})

    return np.concatenate(cell_index), np.concatenate(labels), pd.DataFrame(rows)


@dataclass
class ShardResult:
    context: str
    shard: str
    out_dir: Path
    n_rows_planned: int
    n_cells_shard: int
    n_cells_augmented: int
    n_control_cells: int
    chunk_files: list[Path]
    seconds: float


def compute_se_for_shard(
    shard_path: str | Path,
    controls_path: str | Path,
    context: str,
    out_dir: str | Path,
    perturbation_key: str = "target_gene",
    control_label: str = "non-targeting",
    n_subsamples: int = 6,
    min_cells_to_subsample: int = MIN_CELLS_TO_SUBSAMPLE,
    spacing: str = "linear",
    seed: int = 0,
    min_cells: int = 20,
    chunk_perts: int | None = None,
    layer: str | None = None,
    keep_combined: bool = False,
    scheme: str = "nested",
    replicate_cells: int = 100,
    python: str | None = None,
) -> ShardResult:
    """Augment one shard, attach the control subset, run ``ComputeSE.py``.

    Produces **one chunk per shard**: every label of the shard's 100
    perturbations -- each main row plus its subsamples -- for every response
    gene, in a single ``chunk_000000.csv``.

    ``chunk_perts=None`` (the default) sets ``--chunk-perts`` to the number of
    planned labels so ``ComputeSE.py`` writes exactly one file. Leaving it at
    100 would slice the shard into ~11 arbitrary files and split perturbations
    across them, since a perturbation now contributes ``1 + n_subsamples``
    labels rather than one.

    Stops after step 1 of the pipeline: the output holds ``mean_diff`` and
    ``se``. Running ``ashr`` is a separate decision.
    """
    import anndata as ad

    shard_path, controls_path = Path(shard_path), Path(controls_path)
    python = python or sys.executable
    started = time.time()

    shard_dir = Path(out_dir) / context / shard_path.stem
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard = ad.read_h5ad(shard_path)
    perturbations = shard.obs[perturbation_key].astype(str).to_numpy()
    if (perturbations == control_label).any():
        raise ValueError(
            f"{shard_path.name} contains {control_label!r} cells; shards are "
            "expected to hold only perturbed cells"
        )

    cell_index, labels, plan = plan_shard_rows(
        perturbations,
        n_subsamples=n_subsamples,
        min_cells_to_subsample=min_cells_to_subsample,
        spacing=spacing,
        seed=context_seed(context, seed),
        scheme=scheme,
        replicate_cells=replicate_cells,
    )
    augmented = shard[cell_index].copy()
    augmented.obs = pd.DataFrame(
        {LABEL_KEY: pd.Categorical(labels)},
        index=[f"c{i:09d}" for i in range(len(labels))],
    )
    n_cells_shard, n_cells_augmented = shard.n_obs, augmented.n_obs
    del shard

    controls = ad.read_h5ad(controls_path)
    controls.obs = pd.DataFrame(
        {LABEL_KEY: pd.Categorical([control_label] * controls.n_obs)},
        index=[f"n{i:09d}" for i in range(controls.n_obs)],
    )
    n_control_cells = controls.n_obs

    combined = ad.concat([augmented, controls], axis=0, join="inner")
    del augmented, controls

    combined_path = shard_dir / "combined.h5ad"
    combined.write_h5ad(combined_path)
    del combined

    # One file per shard: cover every label in a single chunk.
    effective_chunk_perts = chunk_perts if chunk_perts else max(len(plan), 1)
    cmd = [python, str(COMPUTE_SE),
           "--adata", str(combined_path),
           "--out-dir", str(shard_dir),
           "--group-key", LABEL_KEY,
           "--control-label", control_label,
           "--min-cells", str(min_cells),
           "--chunk-perts", str(effective_chunk_perts)]
    if layer:
        cmd += ["--layer", layer]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not keep_combined:
        combined_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ComputeSE.py failed on {shard_path.name} (exit {proc.returncode})\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )

    plan["context"] = context
    plan["shard"] = shard_path.stem
    plan.to_csv(shard_dir / "row_plan.csv", index=False)

    return ShardResult(
        context=context,
        shard=shard_path.stem,
        out_dir=shard_dir,
        n_rows_planned=len(plan),
        n_cells_shard=n_cells_shard,
        n_cells_augmented=n_cells_augmented,
        n_control_cells=n_control_cells,
        chunk_files=sorted(shard_dir.glob("chunk_*.csv")),
        seconds=round(time.time() - started, 1),
    )
