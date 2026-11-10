#!/usr/bin/env python3
"""Compute the gene list to keep, from the real matrices, and save it.

Decided on real data only. The column-shuffled augmented matrices have the same
per-gene marginals by construction, so they would give identical counts, and the
permuted-label tables hold no expression values at all.

Save it rather than recomputing per fit: the counts depend on which contexts
exist at the time, a later context can only raise a gene's count, and two fits
run either side of that would silently use different gene sets.

    python scripts/build_gene_list.py --min-affected 100 --threshold 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptex import paths
from cryptex.data import compute_gene_list, filter_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrices", type=Path, default=paths.matrices())
    ap.add_argument("--contexts", nargs="+", default=None)
    ap.add_argument("--min-affected", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="|shrunken logFC| above which a gene counts as "
                         "affected in a (perturbation, context) pair")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/gene_lists/affected_N<N>_thr<t>.tsv")
    args = ap.parse_args()

    out = args.out or (
        paths.root() / "gene_lists"
        / f"affected_N{args.min_affected}_thr{args.threshold:g}.tsv"
    )
    table = compute_gene_list(
        args.matrices,
        min_affected=args.min_affected,
        threshold=args.threshold,
        contexts=args.contexts,
        out_path=out,
    )
    kept = int(table.keep.sum())
    print(f"\ncontexts: {table.attrs['contexts']}")
    print(f"pairs:    {table.attrs['n_pairs']:,}")
    print(f"kept:     {kept:,} of {len(table):,} genes "
          f"({kept/len(table):.1%}) at N>={args.min_affected}, "
          f"|logFC|>{args.threshold}")
    print(f"\n{filter_summary(table.set_index('gene')['n_affected']).to_string(index=False)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
