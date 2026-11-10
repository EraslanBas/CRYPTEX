#!/usr/bin/env python3
"""Stage 1 only: per-shard Welch mean_diff + se via ComputeSE.py.

Uses the SPLIT layout (100 perturbations per shard, controls held separately),
which keeps peak memory around 40-60 GB per shard instead of ~960 GB for a
whole augmented drug. Stops before ashr -- inspect the chunk_*.csv first.

    # sample and persist the 100K control subset, then run every shard
    python scripts/compute_se_shards.py \
        --screen-dir /processed_datasets/VCI/ChemoGenetic_H1_Basak \
        --out-dir    /large_storage/ctc/<user>/ModuleFinder/computese \
        --contexts Stattic --n-control-cells 100000 --shard-jobs 8

    # one shard, to measure first
    python scripts/compute_se_shards.py ... --contexts Stattic --max-shards 1
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptex import paths

import pandas as pd

from cryptex.de import compute_se_for_shard, find_shards, prepare_control_subset

CHEMOGENETIC_CONTEXTS = [
    "DMSO_round2", "DMSO_round2_batch2",
    "AR-A014418", "AZD4573", "Bisindolylmaleimide-I", "CHIR-98014", "DG-172",
    "JTE-607", "LDN-193189", "LY2090314", "Lexibulin", "NSC95397", "PP121",
    "Romidepsin", "Stattic", "VX-11e",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--screen-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=paths.computese())
    p.add_argument("--contexts", nargs="+", default=CHEMOGENETIC_CONTEXTS)
    p.add_argument("--perturbation-key", default="target_gene")
    p.add_argument("--control-label", default="non-targeting")
    p.add_argument("--layer", default=None)

    p.add_argument("--n-control-cells", type=int, default=100_000)
    p.add_argument("--n-subsamples", type=int, default=10)
    p.add_argument("--spacing", choices=["linear", "log"], default="linear")
    p.add_argument("--min-cells-to-subsample", type=int, default=50)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--chunk-perts", type=int, default=None,
                   help="labels per chunk file; default covers the whole shard "
                        "so ComputeSE.py writes one file per shard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scheme", choices=["nested", "disjoint"], default="nested",
                   help="nested: subsamples of sizes spanning [50, n) (the original "
                        "matrices); disjoint: up to --n-subsamples non-overlapping "
                        "pseudobulks of --replicate-cells cells each")
    p.add_argument("--replicate-cells", type=int, default=100,
                   help="cells per pseudobulk with --scheme disjoint")

    p.add_argument("--shard-jobs", type=int, default=1,
                   help="shards to process concurrently (~43 GB each)")
    p.add_argument("--max-shards", type=int, default=None,
                   help="process only the first N shards per context (for timing)")
    p.add_argument("--keep-combined", action="store_true",
                   help="keep the shard+controls h5ad handed to ComputeSE.py")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summary", type=Path, default=None)
    return p


def _one(job):
    shard, controls, context, kwargs = job
    return compute_se_for_shard(shard, controls, context, **kwargs)


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for context in args.contexts:
        shards, _ = find_shards(args.screen_dir, context)
        # one control draw per context, persisted and reused by every shard
        controls = prepare_control_subset(
            args.screen_dir, context, args.out_dir,
            n_control_cells=args.n_control_cells, seed=args.seed,
        )
        if args.max_shards:
            shards = shards[: args.max_shards]

        kwargs = dict(
            out_dir=args.out_dir,
            perturbation_key=args.perturbation_key,
            control_label=args.control_label,
            n_subsamples=args.n_subsamples,
            min_cells_to_subsample=args.min_cells_to_subsample,
            spacing=args.spacing,
            seed=args.seed,
            min_cells=args.min_cells,
            chunk_perts=args.chunk_perts,
            layer=args.layer,
            keep_combined=args.keep_combined,
            scheme=args.scheme,
            replicate_cells=args.replicate_cells,
        )

        todo = []
        for shard in shards:
            done = args.out_dir / context / shard.stem
            if not args.overwrite and any(done.glob("chunk_*.csv")):
                print(f"[skip] {context}/{shard.stem}", flush=True)
                continue
            todo.append((shard, controls, context, kwargs))

        print(f"[{context}] {len(todo)} of {len(shards)} shards to do, "
              f"{args.shard_jobs} at a time", flush=True)

        def record(r, elapsed):
            rows.append({"context": r.context, "shard": r.shard,
                         "rows_planned": r.n_rows_planned,
                         "cells_shard": r.n_cells_shard,
                         "cells_augmented": r.n_cells_augmented,
                         "control_cells": r.n_control_cells,
                         "chunk_files": len(r.chunk_files),
                         "minutes": round(elapsed / 60, 2)})
            print(f"[ok]   {r.context}/{r.shard}  {r.n_rows_planned} rows, "
                  f"{r.n_cells_shard:,} -> {r.n_cells_augmented:,} cells, "
                  f"{len(r.chunk_files)} chunk file(s), {elapsed/60:.1f} min", flush=True)

        if args.shard_jobs <= 1:
            for job in todo:
                t0 = time.time()
                try:
                    record(_one(job), time.time() - t0)
                except Exception as exc:
                    print(f"[FAIL] {context}/{job[0].stem}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.shard_jobs) as pool:
                futures = {pool.submit(_one, job): (job, time.time()) for job in todo}
                for fut in as_completed(futures):
                    job, t0 = futures[fut]
                    try:
                        record(fut.result(), time.time() - t0)
                    except Exception as exc:
                        print(f"[FAIL] {context}/{job[0].stem}: "
                              f"{type(exc).__name__}: {exc}", flush=True)

    summary = pd.DataFrame(rows)
    if len(summary):
        print("\n" + summary.to_string(index=False))
        print(f"\n{len(summary)} shards, {summary.minutes.sum():.1f} min total, "
              f"{summary.rows_planned.sum():,} rows planned")
    if args.summary and len(summary):
        summary.to_csv(args.summary, index=False)
        print(f"summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
