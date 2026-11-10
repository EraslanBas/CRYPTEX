"""Assemble per-context (label x gene) matrices of ashr-shrunken logFC.

Each perturbation contributes 11 rows: its main estimate plus one per
subsample. Rows are therefore *labels* (``<pert>`` and ``<pert>__sub<k>``);
the companion metadata says which perturbation and how many cells produced
each row, so reducing to one row per perturbation is
``meta.variant == "main"``.

Reads, per shard, the ``PosteriorMean`` column of the ashr output and the
label/gene/cell-count columns of the chunk it came from. Those align row for
row, which is asserted here rather than assumed.

    from cryptex.de import build_all_matrices
    summary = build_all_matrices(computese_dir, out_dir)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pv

from .. import paths


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """pyarrow CSV read -- roughly 60x faster than pandas on these files."""
    return pv.read_csv(
        path, convert_options=pv.ConvertOptions(include_columns=columns)
    ).to_pandas()


def load_shard_matrix(shard_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """``(values [labels x genes], row metadata, gene names)`` for one shard."""
    meta = read_columns(shard_dir / "chunk_000000.csv",
                        ["perturbation", "gene", "n_pert", "n_ctrl"])
    post = read_columns(shard_dir / "AshrResult_chunk_000000.csv",
                        ["betahat", "PosteriorMean"])
    if len(meta) != len(post):
        raise ValueError(
            f"{shard_dir.name}: {len(meta)} chunk rows vs {len(post)} ashr rows"
        )

    labels = meta.perturbation.to_numpy()
    n_labels = len(pd.unique(labels))
    n_genes, remainder = divmod(len(meta), n_labels)
    if remainder:
        raise ValueError(
            f"{shard_dir.name}: {len(meta)} rows is not {n_labels} labels x n_genes"
        )

    # ComputeSE.py emits every gene for one label before moving to the next, so
    # the table is label-major / gene-minor and reshapes directly. Verified, not
    # trusted: a silent reordering here would transpose signal across genes.
    genes = meta.gene.to_numpy()[:n_genes]
    gene_block = meta.gene.to_numpy().reshape(n_labels, n_genes)
    if not (gene_block == genes).all():
        raise ValueError(f"{shard_dir.name}: gene order differs between labels")
    label_block = labels.reshape(n_labels, n_genes)
    if not (label_block == label_block[:, :1]).all():
        raise ValueError(f"{shard_dir.name}: labels are not contiguous blocks")

    values = post.PosteriorMean.to_numpy(dtype=np.float32).reshape(n_labels, n_genes)
    row_labels = label_block[:, 0]
    rows = pd.DataFrame({
        "label": row_labels,
        "perturbation": pd.Series(row_labels).str.split("__sub").str[0].to_numpy(),
        "n_cells": meta.n_pert.to_numpy().reshape(n_labels, n_genes)[:, 0],
        "n_ctrl": meta.n_ctrl.to_numpy().reshape(n_labels, n_genes)[:, 0],
        "shard": shard_dir.name,
    })
    rows["variant"] = np.where(rows.label.str.contains("__sub"), "subsample", "main")
    rows = rows[["label", "perturbation", "variant", "n_cells", "n_ctrl", "shard"]]
    return values, rows, genes


def build_context(context_dir: str | Path, out_dir: str | Path,
                  fmt: str = "parquet") -> dict:
    """Stack every finished shard of one context into a single matrix.

    Written **incrementally**, one parquet row group per shard: only one shard
    block (~65 MB) is resident at a time rather than all 24 plus the doubled
    peak of a final ``np.vstack``. An earlier all-at-once version was killed
    mid-context under memory pressure from concurrent ashr jobs.

    The label is stored as a *column*, not an index, because the incremental
    writer needs a fixed schema -- read back with
    ``pd.read_parquet(path).set_index("label")``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    context_dir, out_dir = Path(context_dir), Path(out_dir)
    started = time.time()
    shards = sorted(d for d in context_dir.glob("ad_*")
                    if (d / "AshrResult_chunk_000000.csv").exists())
    if not shards:
        raise FileNotFoundError(f"no completed shards under {context_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    name = context_dir.name
    path = out_dir / f"{name}_PosteriorMean.{fmt}"
    tmp = path.with_suffix(path.suffix + ".tmp")

    writer, schema, genes_ref, metas, n_rows = None, None, None, [], 0
    try:
        for d in shards:
            values, rows, genes = load_shard_matrix(d)
            if genes_ref is None:
                genes_ref = genes
                schema = pa.schema([pa.field("label", pa.string())]
                                   + [pa.field(str(g), pa.float32()) for g in genes])
                writer = pq.ParquetWriter(tmp, schema)
            elif not np.array_equal(genes, genes_ref):
                raise ValueError(f"{d.name}: gene axis differs from {shards[0].name}")

            block = pd.DataFrame(values, columns=[str(g) for g in genes_ref])
            block.insert(0, "label", rows.label.to_numpy())
            writer.write_table(pa.Table.from_pandas(block, schema=schema,
                                                    preserve_index=False))
            metas.append(rows)
            n_rows += len(block)
            del values, block
            print(f"    {d.name}: {len(rows):>5} labels x {len(genes_ref):,} genes",
                  flush=True)
    finally:
        if writer is not None:
            writer.close()

    meta = pd.concat(metas, ignore_index=True)
    # The model consumes rows from several contexts at once, so each row must
    # carry its (perturbation, context) identity -- and `label` alone collides
    # across contexts (every context has its own `A1BG__sub00`).
    meta.insert(0, "row_id", name + "|" + meta.label)
    meta.insert(2, "context", name)
    if meta.label.duplicated().any():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: duplicate labels across shards")

    tmp.replace(path)                      # commit only once complete
    meta.to_csv(out_dir / f"{name}_row_metadata.csv", index=False)

    return {"context": name, "shards": len(shards),
            "rows": n_rows, "genes": len(genes_ref),
            "perturbations": meta.perturbation.nunique(),
            "main_rows": int((meta.variant == "main").sum()),
            "subsample_rows": int((meta.variant == "subsample").sum()),
            "size_GB": round(path.stat().st_size / 1e9, 2),
            "minutes": round((time.time() - started) / 60, 1)}


def build_all_matrices(computese_dir: str | Path, out_dir: str | Path,
                       contexts: list[str] | None = None,
                       fmt: str = "parquet",
                       overwrite: bool = False) -> pd.DataFrame:
    """Build one matrix per context; returns a summary table.

    ``contexts=None`` picks every context whose shards all have an ashr result,
    so a screen still being processed yields only its finished parts.
    """
    computese_dir, out_dir = Path(computese_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if contexts:
        targets = [computese_dir / c for c in contexts]
    else:
        targets = []
        for d in sorted(computese_dir.iterdir()):
            if not d.is_dir() or d.name == "controls":
                continue
            shards = list(d.glob("ad_*/chunk_000000.csv"))
            done = list(d.glob("ad_*/AshrResult_chunk_000000.csv"))
            if shards and len(shards) == len(done):
                targets.append(d)

    summary = []
    for c in targets:
        if (out_dir / f"{c.name}_PosteriorMean.{fmt}").exists() and not overwrite:
            print(f"[skip] {c.name} (exists)", flush=True)
            continue
        print(f"[{c.name}]", flush=True)
        try:
            summary.append(build_context(c, out_dir, fmt))
            print(f"  -> {summary[-1]}", flush=True)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    return pd.DataFrame(summary)


def load_matrices(
    matrices_dir: str | Path | None = None,
    contexts: list[str] | None = None,
    variant: str | None = None,
    fmt: str = "parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load contexts and stack them into one ``(rows x genes)`` design matrix.

    This is what the model consumes. Every row carries its
    ``(perturbation, context)`` identity in the returned metadata, indexed by a
    globally unique ``row_id`` of ``"<context>|<label>"`` -- ``label`` alone is
    not unique, since every context has its own ``A1BG__sub00``.

    Parameters
    ----------
    variant
        ``"main"`` keeps one row per (perturbation, context); ``"subsample"``
        keeps only the subsampled rows; ``None`` keeps all 11 per perturbation.

    Returns
    -------
    (X, meta)
        ``X`` is [rows x genes] float32 indexed by ``row_id``; ``meta`` has
        ``row_id, label, context, perturbation, variant, n_cells, n_ctrl,
        shard`` in the same order.
    """
    if matrices_dir is None:
        matrices_dir = paths.matrices()
    matrices_dir = Path(matrices_dir)
    found = sorted(matrices_dir.glob(f"*_PosteriorMean.{fmt}"))
    names = [f.name.replace(f"_PosteriorMean.{fmt}", "") for f in found]
    if contexts:
        missing = set(contexts) - set(names)
        if missing:
            raise FileNotFoundError(f"no matrix for {sorted(missing)} in {matrices_dir}")
        names = [n for n in names if n in set(contexts)]

    blocks, metas = [], []
    for name in names:
        meta = pd.read_csv(matrices_dir / f"{name}_row_metadata.csv")
        if "context" not in meta.columns:       # built before context was recorded
            raise ValueError(
                f"{name}_row_metadata.csv predates the context column; rebuild it "
                "with build_all_matrices(..., overwrite=True)"
            )
        block = pd.read_parquet(matrices_dir / f"{name}_PosteriorMean.{fmt}")
        if "label" in block.columns:
            block = block.set_index("label")
        block.index = meta.row_id.to_numpy()

        if variant is not None:
            keep = (meta.variant == variant).to_numpy()
            block, meta = block[keep], meta[keep]
        blocks.append(block)
        metas.append(meta)
        print(f"  {name}: {block.shape[0]:,} rows", flush=True)

    X = pd.concat(blocks, axis=0)
    meta = pd.concat(metas, ignore_index=True).set_index("row_id", drop=False)
    X.index.name = "row_id"
    if not X.index.equals(meta.index):
        raise ValueError("matrix and metadata row order disagree")
    return X, meta
