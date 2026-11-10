#!/usr/bin/env python3
"""
Pivot ashr results from long → wide and concatenate across chunks.

Reads every AshrResult_chunk_*.csv in a directory (each long-format with
columns ``perturbation``, ``gene``, ``PosteriorMean``, ...), pivots each into
a perturbation × gene wide table of posterior means, and row-concatenates
them into one giant matrix. Output is a single CSV or parquet file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


ASHR_RE = re.compile(r"^AshrResult_chunk_\d{6}\.csv$")


def find_chunk_files(d: Path) -> list[Path]:
    files = [p for p in d.iterdir() if p.is_file() and ASHR_RE.match(p.name)]
    return sorted(files, key=lambda p: p.name)


def pick_value_col(df: pd.DataFrame) -> str:
    candidates = ["PosteriorMean", "ash_postmean", "posterior_mean", "PosteriorMeanEstimate"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a PosteriorMean column. Available columns: {list(df.columns)[:30]} ..."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Read AshrResult_chunk_*.csv files, pivot each into a perturbation × gene "
            "matrix of PosteriorMean, row-concat into one giant matrix, and write a "
            "single output file."
        )
    )
    ap.add_argument("--dir", type=Path, required=True,
                     help="Directory containing AshrResult_chunk_*.csv")
    ap.add_argument("--out", type=Path, required=True,
                     help="Output path for the giant matrix (csv or parquet)")
    ap.add_argument("--format", choices=["csv", "parquet"], default="parquet",
                     help="Output format (default: parquet — much smaller for wide matrices)")
    ap.add_argument("--delete-inputs", action="store_true",
                     help="Delete AshrResult_chunk_*.csv after a successful write.")
    args = ap.parse_args()

    d = args.dir
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")

    files = find_chunk_files(d)
    if not files:
        raise RuntimeError(f"No files matching AshrResult_chunk_*.csv found in {d}")

    mats: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)

        required = {"perturbation", "gene"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{f.name} missing required columns: {sorted(missing)}")

        value_col = pick_value_col(df)
        wide = df.pivot_table(
            index="perturbation",
            columns="gene",
            values=value_col,
            aggfunc="first",
        )
        wide.index.name = "perturbation"
        mats.append(wide)
        print(f"[ok] {f.name}: {wide.shape[0]} perts × {wide.shape[1]} genes")

    giant = pd.concat(mats, axis=0, join="outer", sort=False)

    if giant.index.duplicated().any():
        ndup = int(giant.index.duplicated().sum())
        print(f"[warn] {ndup} duplicated perturbation rows; keeping first occurrence.")
        giant = giant[~giant.index.duplicated(keep="first")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        giant.to_parquet(args.out)
    else:
        giant.to_csv(args.out)

    print(f"[done] wrote giant matrix: {args.out}  shape={giant.shape}")

    if args.delete_inputs:
        for f in files:
            f.unlink()
        print(f"[clean] deleted {len(files)} AshrResult_chunk_*.csv files")


if __name__ == "__main__":
    main()
