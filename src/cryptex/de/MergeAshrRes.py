#!/usr/bin/env python3
"""
Merge per-chunk metadata back into the ashr output.

For each pair (chunk_NNNNNN.csv, AshrResult_chunk_NNNNNN.csv) in a directory,
this script:

1. Verifies row counts match.
2. Drops ``mean_diff`` and ``se`` from the chunk file (already represented in
   the ashr output) and merges every other chunk column (``perturbation``,
   ``gene``, ``mean_pert``, ``mean_ctrl``, ``n_pert``, ``n_ctrl``, ...) into
   the ashr file by row order. The ashr file is overwritten with the merged
   frame.
3. Optionally row-binds every merged ``AshrResult_chunk_*.csv`` into a single
   consolidated ``AshrResult_all_chunks.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

ASHR_RE = re.compile(r"^AshrResult_chunk_(\d{6})\.csv$")


def merge_one_pair(chunk_dir: Path, chunk_id: str, drop_cols: set[str]) -> Path | None:
    ashr_path  = chunk_dir / f"AshrResult_chunk_{chunk_id}.csv"
    chunk_path = chunk_dir / f"chunk_{chunk_id}.csv"

    if not ashr_path.exists():
        print(f"[skip] missing ASHR file: {ashr_path}")
        return None
    if not chunk_path.exists():
        print(f"[skip] missing chunk file: {chunk_path}")
        return None

    ashr  = pd.read_csv(ashr_path)
    chunk = pd.read_csv(chunk_path)

    if len(ashr) != len(chunk):
        raise ValueError(
            f"Row count mismatch for chunk_{chunk_id}: "
            f"{ashr_path.name} has {len(ashr):,} rows, "
            f"{chunk_path.name} has {len(chunk):,} rows."
        )

    chunk_keep = chunk.drop(
        columns=[c for c in drop_cols if c in chunk.columns],
        errors="ignore",
    )
    cols_to_add = [c for c in chunk_keep.columns if c not in ashr.columns]
    chunk_keep  = chunk_keep[cols_to_add]

    merged = pd.concat(
        [
            ashr.reset_index(drop=True),
            chunk_keep.reset_index(drop=True),
        ],
        axis=1,
    )

    merged.to_csv(ashr_path, index=False)
    print(f"[ok] merged chunk_{chunk_id} (+{len(cols_to_add)} cols)")
    return ashr_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Merge chunk metadata into ASHR results by row order, overwrite ASHR files, "
            "then row-bind all ASHR files into one CSV."
        )
    )
    ap.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing chunk_*.csv and AshrResult_chunk_*.csv",
    )
    ap.add_argument(
        "--drop-cols",
        type=str,
        default="mean_diff,se",
        help="Comma-separated column names to exclude from chunk files (default: mean_diff,se)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for combined CSV (default: <dir>/AshrResult_all_chunks.csv)",
    )
    ap.add_argument(
        "--no-combine",
        action="store_true",
        help="Skip the final row-bind into AshrResult_all_chunks.csv.",
    )
    args = ap.parse_args()

    chunk_dir = args.dir
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {chunk_dir}")

    drop_cols = {c.strip() for c in args.drop_cols.split(",") if c.strip()}

    ashr_files = sorted(
        p for p in chunk_dir.iterdir()
        if p.is_file() and ASHR_RE.match(p.name)
    )
    if not ashr_files:
        raise RuntimeError(f"No AshrResult_chunk_*.csv files found in {chunk_dir}")

    # ---- Step 1: merge per-chunk metadata ----
    processed_files: list[Path] = []
    for p in ashr_files:
        chunk_id = ASHR_RE.match(p.name).group(1)
        out = merge_one_pair(chunk_dir, chunk_id, drop_cols)
        if out is not None:
            processed_files.append(out)

    if not processed_files:
        raise RuntimeError("No ASHR files were successfully processed.")

    # ---- Step 2: row-bind all ASHR results ----
    if args.no_combine:
        print("[done] skipped row-bind (--no-combine).")
        return

    combined = pd.concat(
        (pd.read_csv(p) for p in processed_files),
        ignore_index=True,
    )
    out_path = args.out or (chunk_dir / "AshrResult_all_chunks.csv")
    combined.to_csv(out_path, index=False)
    print(f"[done] wrote combined file: {out_path} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
