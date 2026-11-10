"""One call: augmented AnnData -> shrunken logFC matrix + per-row cell counts.

Chains input construction (``arms.py``) through the unmodified
ChemoGeneticScreens pipeline (``RunAshrPipeline.sh``: ``ComputeSE.py`` ->
``run_ashr_on_chunk.R`` -> ``MergeAshrRes.py`` -> ``AshrGenerateMatrix.py``).

The resulting matrix carries **both** the full-data and the subsampled
estimates as separate rows, indexed by label (``<pert>`` and
``<pert>__sub<k>``), and the companion ``row_metadata.csv`` records how many
cells went into each row.
"""

from __future__ import annotations

import collections
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .arms import (
    MIN_CELLS_TO_SUBSAMPLE,
    N_CONTROL_CELLS,
    SUBSAMPLE_SEP,
    build_augmented_adata,
    subsample_sizes,
)

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "RunAshrPipeline.sh"


@dataclass
class PosteriorRun:
    """Everything one run produced."""

    name: str
    out_dir: Path
    matrix_path: Path
    row_metadata_path: Path
    augmented_h5ad: Path | None
    matrix: pd.DataFrame
    row_metadata: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - display only
        n_full = int((self.row_metadata["variant"] == "full").sum())
        n_sub = int((self.row_metadata["variant"] == "subsample").sum())
        return (
            f"PosteriorRun({self.name!r}: {self.matrix.shape[0]} rows "
            f"({n_full} full + {n_sub} subsampled) x {self.matrix.shape[1]} genes)"
        )


def generate_posterior_matrices(
    adata,
    context: str,
    out_dir: str | Path,
    perturbation_key: str = "target_gene",
    control_label: str = "non-targeting",
    n_subsamples: int = 1,
    min_cells_to_subsample: int = MIN_CELLS_TO_SUBSAMPLE,
    n_control_cells: int = N_CONTROL_CELLS,
    spacing: str = "linear",
    seed: int = 0,
    permute: bool = False,
    permute_seed: int = 0,
    layer: str | None = None,
    min_cells: int = 20,
    chunk_perts: int = 100,
    n_ashr_jobs: int = 6,
    matrix_format: str = "csv",
    keep_augmented_h5ad: bool = False,
    python: str | None = None,
    rscript: str = "Rscript",
) -> PosteriorRun:
    """Generate a shrunken-logFC matrix holding full **and** subsampled rows.

    Parameters
    ----------
    adata
        AnnData (in-memory or backed) or a path to one ``.h5ad``. One context
        per call -- the ChemoGenetic per-drug files carry no drug column, so
        the context name is supplied here and used to name the outputs.
    context
        Name for this context; the matrix lands at
        ``<out_dir>/<name>/PosteriorMean_matrix_<name>.<fmt>`` where ``name``
        is ``context`` (plus ``_permuted_seed<k>`` for the null arm).
    n_subsamples
        Subsamples requested per eligible perturbation -- the user-defined
        knob. For a perturbation with more than ``min_cells_to_subsample``
        cells, the sizes **span** ``[min_cells_to_subsample, n_cells]``
        (endpoints included) so the requested number covers the range as
        evenly as it can; which cells go into each is still random. Fewer are
        produced if the range cannot supply that many distinct sizes.
        Perturbations at or below the floor contribute only their full row.
    spacing
        ``"linear"`` spreads sizes evenly in ``n``; ``"log"`` spreads them
        geometrically, covering *precision* more evenly since ``se`` scales as
        ``1/sqrt(n)``. See
        :func:`~cryptex.de.arms.subsample_sizes`.
    n_control_cells
        Size of the fixed control set for this context, shared by every row
        (full and subsampled alike). Defaults to 10,000. Controls beyond this
        are dropped, so the ``var_c / n_ctrl`` term of the standard error is
        identical across rows and comparable across contexts.
    permute
        Build the label-permuted null instead: perturbation labels are shuffled
        among perturbed cells, controls untouched, cell counts preserved.
    min_cells, chunk_perts, n_ashr_jobs, layer, matrix_format
        Passed through to ``RunAshrPipeline.sh``. ``chunk_perts`` reproduces
        the original per-100-perturbation ashr grouping; full and subsampled
        rows of the same perturbation may land in different chunks, so raise it
        (or set it above the row count) if you want one shared prior.

    Returns
    -------
    PosteriorRun
        With ``matrix`` (rows = labels) and ``row_metadata`` (``label,
        perturbation, variant, subsample, n_cells_used, n_ctrl``).

    Notes
    -----
    Requires R with ``ashr``. Subsampled cells are duplicated in the augmented
    ``.h5ad`` so that one ``ComputeSE.py`` pass yields both variants, which
    means the temporary file is larger than the input -- roughly
    ``1 + n_subsamples/2`` times the perturbed cells.
    """
    import anndata as ad

    if not PIPELINE.exists():
        raise FileNotFoundError(f"pipeline driver missing: {PIPELINE}")
    python = python or sys.executable

    if isinstance(adata, (str, Path)):
        adata = ad.read_h5ad(str(adata), backed="r")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = context if not permute else f"{context}_permuted_seed{permute_seed}"
    label_key = "mf_label"

    started = time.time()
    augmented, plan = build_augmented_adata(
        adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        label_key=label_key,
        n_subsamples=n_subsamples,
        min_cells_to_subsample=min_cells_to_subsample,
        n_control_cells=n_control_cells,
        spacing=spacing,
        seed=seed,
        permute=permute,
        permute_seed=permute_seed,
    )
    h5ad_path = out_dir / f"{name}.h5ad"
    augmented.write_h5ad(h5ad_path)
    build_seconds = time.time() - started
    n_augmented_cells = int(augmented.n_obs)
    n_input_cells, n_genes = int(adata.n_obs), int(adata.n_vars)

    # Free both objects before the pipeline runs: it reads the augmented file
    # from disk and never touches these again. On a full ChemoGenetic context
    # that is ~240 GB (source) + ~500 GB (augmented) handed back, which is the
    # difference between fitting in a 966 GB allocation and being SIGKILLed.
    del augmented
    adata = None
    gc.collect()

    env = {
        **os.environ,
        "OUT_BASE": str(out_dir),
        "PYTHON": python,
        "RSCRIPT": rscript,
        "GROUP_KEY": label_key,
        "CONTROL_LABEL": control_label,
        "MIN_CELLS": str(min_cells),
        "CHUNK_PERTS": str(chunk_perts),
        "N_ASHR_JOBS": str(n_ashr_jobs),
        "MATRIX_FORMAT": matrix_format,
        "LAYER": layer or "",
    }
    started = time.time()
    proc = subprocess.run(
        ["bash", str(PIPELINE), str(h5ad_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    pipeline_seconds = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(
            "RunAshrPipeline.sh failed "
            f"(exit {proc.returncode}).\n--- stdout ---\n{proc.stdout[-3000:]}"
            f"\n--- stderr ---\n{proc.stderr[-3000:]}"
        )

    run_dir = out_dir / name
    matrix_path = run_dir / f"PosteriorMean_matrix_{name}.{matrix_format}"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"pipeline reported success but {matrix_path} is missing.\n"
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )
    matrix = (
        pd.read_parquet(matrix_path)
        if matrix_format == "parquet"
        else pd.read_csv(matrix_path, index_col=0)
    )

    row_metadata = _collect_row_metadata(run_dir, plan, context)
    row_metadata_path = run_dir / "row_metadata.csv"
    row_metadata.to_csv(row_metadata_path, index=False)

    meta = {
        "name": name,
        "context": context,
        "permuted": permute,
        "permute_seed": permute_seed if permute else None,
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "n_subsamples": n_subsamples,
        "min_cells_to_subsample": min_cells_to_subsample,
        "n_control_cells": n_control_cells,
        "spacing": spacing,
        "seed": seed,
        "min_cells": min_cells,
        "chunk_perts": chunk_perts,
        "layer": layer,
        "n_input_cells": n_input_cells,
        "n_augmented_cells": n_augmented_cells,
        "n_genes": n_genes,
        "n_rows": int(matrix.shape[0]),
        "build_seconds": round(build_seconds, 1),
        "pipeline_seconds": round(pipeline_seconds, 1),
        "ashr_version": _ashr_version(rscript),
    }
    (run_dir / "module_finder_manifest.json").write_text(json.dumps(meta, indent=2))

    if not keep_augmented_h5ad:
        h5ad_path.unlink(missing_ok=True)

    return PosteriorRun(
        name=name,
        out_dir=run_dir,
        matrix_path=matrix_path,
        row_metadata_path=row_metadata_path,
        augmented_h5ad=h5ad_path if keep_augmented_h5ad else None,
        matrix=matrix,
        row_metadata=row_metadata,
        meta=meta,
    )


def context_path(screen_dir: str | Path, context: str) -> Path:
    """Locate a context's ``.h5ad``, nested (``<ctx>/<ctx>.h5ad``) or flat."""
    screen_dir = Path(screen_dir)
    for candidate in (screen_dir / context / f"{context}.h5ad",
                      screen_dir / f"{context}.h5ad"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no .h5ad for context {context!r} under {screen_dir} "
        f"(tried <ctx>/<ctx>.h5ad and <ctx>.h5ad)"
    )


def run_screen(
    screen_dir: str | Path,
    contexts: list[str],
    out_dir: str | Path,
    isolate: bool = True,
    skip_existing: bool = True,
    matrix_format: str = "parquet",
    min_free_tb: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Generate posterior matrices for every context. Returns a summary table.

    Parameters
    ----------
    isolate
        Run each context in a **fresh subprocess** (the default). These objects
        are ~150 GB resident and roughly twice that once augmented; Python and
        anndata do not reliably hand freed memory back to the OS, so a
        long-lived process accumulates peak across contexts and eventually
        dies. A subprocess exit reclaims everything unconditionally, and a
        crash costs one context rather than the whole run. Set False to run
        in-process (fine for small data, or for debugging a traceback).
    skip_existing
        Skip contexts whose matrix is already on disk, making the call
        resumable.
    min_free_tb
        Stop if the output volume drops below this.
    **kwargs
        Passed to :func:`generate_posterior_matrices` -- ``n_subsamples``,
        ``spacing``, ``n_control_cells``, ``min_cells_to_subsample``,
        ``min_cells``, ``chunk_perts``, ``n_ashr_jobs``, ``permute``, ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for context in contexts:
        done = out_dir / context / f"PosteriorMean_matrix_{context}.{matrix_format}"
        if skip_existing and done.exists():
            print(f"[skip] {context:<24} already done", flush=True)
            continue

        started = time.time()
        try:
            if isolate:
                meta = _run_isolated(
                    screen_dir, context, out_dir,
                    matrix_format=matrix_format, **kwargs
                )
            else:
                import anndata as ad

                adata = ad.read_h5ad(context_path(screen_dir, context))
                try:
                    result = generate_posterior_matrices(
                        adata, context=context, out_dir=out_dir,
                        matrix_format=matrix_format, **kwargs
                    )
                    meta = result.meta
                finally:
                    del adata
                    gc.collect()
            minutes = (time.time() - started) / 60
            rows.append({
                "context": context, "rows": meta["n_rows"], "genes": meta["n_genes"],
                "cells_in": meta["n_input_cells"],
                "cells_augmented": meta["n_augmented_cells"],
                "minutes": round(minutes, 1), "status": "ok",
            })
            print(f"[ok]   {context:<24} {meta['n_rows']:,} rows x "
                  f"{meta['n_genes']:,} genes in {minutes:.1f} min", flush=True)
        except Exception as exc:
            rows.append({"context": context, "status": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {context:<24} {type(exc).__name__}: {exc}", flush=True)

        free_tb = shutil.disk_usage(out_dir).free / 1e12
        print(f"       {free_tb:.1f} TB free", flush=True)
        if free_tb < min_free_tb:
            print(f"       stopping: under {min_free_tb} TB left", flush=True)
            break

    return pd.DataFrame(rows)


def _run_isolated(screen_dir, context, out_dir, **kwargs) -> dict:
    """Run one context via the CLI in a subprocess; return its manifest."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_posterior_matrices.py"
    if not script.exists():
        raise FileNotFoundError(f"driver script not found: {script}")

    cmd = [sys.executable, str(script),
           "--screen-dir", str(screen_dir),
           "--out-dir", str(out_dir),
           "--contexts", context,
           "--no-isolate"]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd += [flag, str(value)]

    # Stream the child's output rather than capturing it: a context takes hours,
    # and a silent log for that long is indistinguishable from a hang. Keep a
    # tail so a failure still reports something useful.
    tail: collections.deque[str] = collections.deque(maxlen=40)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        print(f"       | {line}", flush=True)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"subprocess failed (exit {returncode})\n" + "\n".join(tail)
        )
    name = context if not kwargs.get("permute") else \
        f"{context}_permuted_seed{kwargs.get('permute_seed', 0)}"
    manifest = out_dir / name / "module_finder_manifest.json"
    return json.loads(manifest.read_text())


def _collect_row_metadata(
    run_dir: Path, plan: pd.DataFrame, context: str
) -> pd.DataFrame:
    """Per-row cell counts, taken from what the pipeline itself recorded.

    ``ComputeSE.py`` writes ``n_pert``/``n_ctrl`` into every chunk and
    ``MergeAshrRes.py`` merges them into the ashr output, so the counts used
    for each estimate come straight from the pipeline rather than being
    recomputed here.
    """
    frames = []
    for path in sorted(run_dir.glob("AshrResult_chunk_*.csv")):
        frames.append(
            pd.read_csv(path, usecols=["perturbation", "n_pert", "n_ctrl"])
            .drop_duplicates()
        )
    if not frames:
        raise FileNotFoundError(f"no AshrResult_chunk_*.csv under {run_dir}")

    counts = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .rename(columns={"perturbation": "label", "n_pert": "n_cells_used"})
    )
    merged = plan.merge(counts, on="label", how="right")
    merged["context"] = context
    missing = merged["variant"].isna()
    if missing.any():  # label present in output but not in the plan
        merged.loc[missing, "variant"] = "unknown"
    return merged[
        [
            "label",
            "perturbation",
            "context",
            "variant",
            "subsample",
            "n_cells_used",
            "n_cells_planned",
            "n_ctrl",
        ]
    ].sort_values(["perturbation", "variant", "subsample"], na_position="first")


def _ashr_version(rscript: str) -> str | None:
    if shutil.which(rscript) is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", 'cat(as.character(packageVersion("ashr")))'],
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip() or None
