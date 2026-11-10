#!/usr/bin/env python3
"""
Compute per-perturbation mean difference (pert - ctrl) and SE in expression space,
writing one output file per chunk of perturbations (CSV).

Example:
  python compute_log_mean_diff_chunks.py \
    --adata path/to/data.h5ad \
    --out-dir results/ \
    --group-key target_gene \
    --control-label non-targeting \
    --layer log1p \
    --min-cells 20 \
    --chunk-perts 100

Outputs:
  results/chunk_000000.csv
  results/chunk_000100.csv
  ...
  results/manifest.tsv (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _sum_and_sumsq(X) -> Tuple[np.ndarray, np.ndarray]:
    """Column-wise sum and sum of squares for dense or sparse matrices."""
    if sp.issparse(X):
        X = X.tocsr()
        s = np.asarray(X.sum(axis=0)).ravel().astype(np.float64, copy=False)
        ss = np.asarray(X.multiply(X).sum(axis=0)).ravel().astype(np.float64, copy=False)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        s = Xd.sum(axis=0)
        ss = (Xd * Xd).sum(axis=0)
    return s, ss


def _mean_and_var_from_sum(s: np.ndarray, ss: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """From sum and sumsq compute mean and unbiased variance (ddof=1)."""
    mean = s / n
    if n < 2:
        var = np.zeros_like(mean, dtype=np.float64)
        return mean, var

    ex2 = ss / n
    var_pop = np.maximum(ex2 - mean * mean, 0.0)
    var = var_pop * (n / (n - 1))
    return mean, var


def _validate_and_load_adata(adata_path: Path):
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(
            "This script requires `anndata` (and typically `scanpy`). "
            "Install with: pip install anndata scanpy"
        ) from e

    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    return ad.read_h5ad(str(adata_path))


def compute_and_write_chunks(
    adata_path: Path,
    out_dir: Path,
    *,
    group_key: str = "target_gene",
    control_label: str = "non-targeting",
    layer: Optional[str] = None,
    min_cells: int = 20,
    chunk_perts: int = 100,
    write_manifest: bool = True,
    consolidate: bool = False,
) -> List[Path]:
    """
    Writes one CSV file per chunk. Returns list of chunk file paths.
    Optionally writes a manifest.tsv and/or a consolidated CSV.
    """
    adata = _validate_and_load_adata(adata_path)

    if group_key not in adata.obs:
        raise KeyError(f"--group-key '{group_key}' not found in adata.obs")

    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.tocsr()

    groups = adata.obs[group_key].astype("string").to_numpy()
    genes = adata.var_names.to_numpy()

    ctrl_mask = groups == control_label
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl < 2:
        raise ValueError(f"Need >=2 control cells for '{control_label}', found {n_ctrl}.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- control stats once ----
    s_c, ss_c = _sum_and_sumsq(X[ctrl_mask])
    mean_c, var_c = _mean_and_var_from_sum(s_c, ss_c, n_ctrl)

    # ---- perturbations to process ----
    pert_counts = pd.Series(groups).value_counts()
    perts = [p for p in np.unique(groups) if p != control_label]
    perts = [p for p in perts if int(pert_counts.get(p, 0)) >= min_cells]

    if not perts:
        raise ValueError(
            f"No perturbations passed filters (min_cells={min_cells}) excluding control '{control_label}'."
        )

    chunk_files: List[Path] = []

    # ---- process in chunks ----
    for start in range(0, len(perts), chunk_perts):
        batch = perts[start : start + chunk_perts]

        rows = []
        for pert in batch:
            pert_mask = groups == pert
            n_p = int(pert_mask.sum())
            if n_p < 2:
                continue

            s_p, ss_p = _sum_and_sumsq(X[pert_mask])
            mean_p, var_p = _mean_and_var_from_sum(s_p, ss_p, n_p)

            mean_diff = mean_p - mean_c
            se = np.sqrt(var_p / n_p + var_c / n_ctrl)

            df = pd.DataFrame(
                {
                    "perturbation": pert,
                    "gene": genes,
                    "mean_pert": mean_p,
                    "mean_ctrl": mean_c,  # identical for all perts; kept for convenience
                    "mean_diff": mean_diff,
                    "se": se,
                    "n_pert": n_p,
                    "n_ctrl": n_ctrl,
                }
            )
            rows.append(df)

        chunk_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        chunk_path = out_dir / f"chunk_{start:06d}.csv"
        chunk_df.to_csv(chunk_path, index=False)
       
    return 


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute mean differences and SEs per perturbation vs control; write one CSV per chunk."
    )
    p.add_argument("--adata", required=True, type=Path, help="Path to input .h5ad")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory to write chunk CSV files")
    p.add_argument("--group-key", default="target_gene", help="adata.obs column holding perturbation labels")
    p.add_argument("--control-label", default="non-targeting", help="Label in group-key that denotes controls")
    p.add_argument("--layer", default=None, help="Layer name to use (default: adata.X)")
    p.add_argument("--min-cells", type=int, default=20, help="Minimum cells required per perturbation")
    p.add_argument("--chunk-perts", type=int, default=100, help="Number of perturbations per chunk file")
    p.add_argument("--no-manifest", action="store_true", help="Do not write manifest.tsv")
    p.add_argument(
        "--consolidate",
        action="store_true",
        help="Also write a single consolidated CSV (all_chunks.csv). Can be large.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    compute_and_write_chunks(
        adata_path=args.adata,
        out_dir=args.out_dir,
        group_key=args.group_key,
        control_label=args.control_label,
        layer=args.layer,
        min_cells=args.min_cells,
        chunk_perts=args.chunk_perts,
        write_manifest=not args.no_manifest,
        consolidate=args.consolidate,
    )


if __name__ == "__main__":
    main()
