#!/usr/bin/env python3
"""Generate shrunken-logFC (posterior-mean) matrices for a whole screen.

Prefer this over a notebook for anything large. These objects are ~150 GB
resident and roughly twice that once augmented, and Python does not reliably
return freed memory to the OS -- a long-lived kernel accumulates peak until it
dies, taking the run with it. By default each context runs in a fresh
subprocess, so its memory is reclaimed unconditionally on exit and a crash
costs one context instead of everything.

    # whole screen, one subprocess per context, resumable
    python scripts/build_posterior_matrices.py \
        --screen-dir /processed_datasets/VCI/ChemoGenetic_H1_Basak \
        --out-dir /large_storage/ctc/beraslan/ModuleFinder/posterior_matrices \
        --n-subsamples 6

    # one context, in this process (what --isolate spawns internally)
    python scripts/build_posterior_matrices.py ... --contexts Stattic --no-isolate

Requires R with the `ashr` package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptex import paths

from cryptex.de import context_path, generate_posterior_matrices, run_screen

#: The 16 contexts of the chemogenetic screen. Listed rather than globbed: the
#: source directory also holds Round-1 contexts (CHIR, DMSO, KYA, LDN, PFI1,
#: RGFP) with the same file layout.
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
    p.add_argument("--out-dir", type=Path, default=paths.root())
    p.add_argument("--contexts", nargs="+", default=CHEMOGENETIC_CONTEXTS,
                   help="default: the 16 chemogenetic contexts")

    p.add_argument("--perturbation-key", default="target_gene")
    p.add_argument("--control-label", default="non-targeting")
    p.add_argument("--layer", default=None)

    p.add_argument("--n-subsamples", type=int, default=6)
    p.add_argument("--spacing", choices=["linear", "log"], default="linear")
    p.add_argument("--min-cells-to-subsample", type=int, default=50)
    p.add_argument("--n-control-cells", type=int, default=10_000)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--chunk-perts", type=int, default=100)
    p.add_argument("--n-ashr-jobs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--permute", action="store_true",
                   help="label-permuted null instead of the real arm")
    p.add_argument("--permute-seed", type=int, default=0)

    p.add_argument("--matrix-format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--keep-augmented-h5ad", action="store_true")
    p.add_argument("--no-isolate", dest="isolate", action="store_false",
                   help="run in this process instead of a subprocess per context")
    p.add_argument("--overwrite", dest="skip_existing", action="store_false")
    p.add_argument("--min-free-tb", type=float, default=0.5)
    p.add_argument("--summary", type=Path, default=None,
                   help="write the summary table here as CSV")
    return p


def main() -> int:
    args = build_parser().parse_args()

    shared = dict(
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        layer=args.layer,
        n_subsamples=args.n_subsamples,
        spacing=args.spacing,
        min_cells_to_subsample=args.min_cells_to_subsample,
        n_control_cells=args.n_control_cells,
        min_cells=args.min_cells,
        chunk_perts=args.chunk_perts,
        n_ashr_jobs=args.n_ashr_jobs,
        seed=args.seed,
        permute=args.permute,
        permute_seed=args.permute_seed,
        keep_augmented_h5ad=args.keep_augmented_h5ad,
    )

    # Single context in-process: this is the leaf the isolated mode spawns.
    if not args.isolate and len(args.contexts) == 1:
        import anndata as ad

        context = args.contexts[0]
        path = context_path(args.screen_dir, context)
        print(f"[load] {path}", flush=True)
        adata = ad.read_h5ad(path)
        result = generate_posterior_matrices(
            adata, context=context, out_dir=args.out_dir,
            matrix_format=args.matrix_format, **shared,
        )
        print(f"[done] {result}", flush=True)
        print(f"       {result.matrix_path}", flush=True)
        return 0

    summary = run_screen(
        screen_dir=args.screen_dir,
        contexts=args.contexts,
        out_dir=args.out_dir,
        isolate=args.isolate,
        skip_existing=args.skip_existing,
        matrix_format=args.matrix_format,
        min_free_tb=args.min_free_tb,
        **shared,
    )

    if len(summary):
        print("\n" + summary.to_string(index=False))
        ok = summary[summary.status == "ok"] if "status" in summary else summary
        if len(ok):
            print(f"\n{len(ok)} contexts, {ok.minutes.sum():.0f} min, "
                  f"{ok.rows.sum():,} rows")
    if args.summary:
        summary.to_csv(args.summary, index=False)
        print(f"summary -> {args.summary}")
    failed = summary[summary.status != "ok"] if len(summary) and "status" in summary else []
    return 1 if len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
