#!/usr/bin/env python3
"""Draw the train/val vs test split once and write both partitions separately.

    python scripts/build_split.py                      # pair_seed0, defaults below
    python scripts/build_split.py --split-seed 1

Test holds out 10% of (perturbation, context) pairs, every perturbation keeping
at least one pair in train/val. The gene list is computed on train/val
full-data rows only (|logFC| > ln 1.2 in more than 150 pairs) and applied to
both partitions; synthetic rows are shuffled over each context's rows before
the split (--shuffle-within context, default) or within (partition, context)
(--shuffle-within partition). See
cryptex.data.splits for the layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptex import paths
from cryptex.data.splits import build_split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrices", type=Path, default=paths.matrices())
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: $CRYPTEX_ROOT/splits/pair_seed<split-seed>")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-affected", type=int, default=151,
                   help="keep genes affected in at least this many train/val "
                        "pairs (151 = 'more than 150')")
    p.add_argument("--threshold", type=float, default=float(np.log(1.2)),
                   help="|logFC| above which a gene counts as affected (ln 1.2)")
    p.add_argument("--shuffle-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--contexts", nargs="+", default=None)
    p.add_argument("--shuffle-within", choices=["context", "partition"],
                   default="context",
                   help="context: shuffle all of a context's rows before the "
                        "split (test values can enter train/val synthetic "
                        "rows); partition: shuffle within partition x context")
    p.add_argument("--keep-perturbed-genes", action="store_true",
                   help="also keep every measured perturbation target as a response "
                        "gene, whether or not it passes the filter")
    p.add_argument("--min-pair-cells", type=int, default=0,
                   help="drop (perturbation, context) pairs whose full-data row has fewer "
                        "cells, from every partition (after the test draw)")
    p.add_argument("--gene-list", type=Path, default=None,
                   help="keep exactly the genes of this gene_list.tsv (keep column), "
                        "instead of the filter")
    args = p.parse_args()

    manifest = build_split(
        matrices_dir=args.matrices,
        out_dir=args.out_dir,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        min_affected=args.min_affected,
        threshold=args.threshold,
        shuffle_seeds=tuple(args.shuffle_seeds),
        keep_perturbed_genes=args.keep_perturbed_genes,
        min_pair_cells=args.min_pair_cells,
        gene_list=(None if args.gene_list is None else
                   __import__("cryptex.data.gene_filter", fromlist=["load_gene_list"]).load_gene_list(args.gene_list)),
        contexts=args.contexts,
        shuffle_within=args.shuffle_within,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
