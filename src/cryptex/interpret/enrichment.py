"""Export ranked gene lists for pathway over-representation analysis.

Deliberately thin: this writes the ranked lists and defers to the existing KEGG
ORA pipeline in ``ChemoGeneticScreens`` rather than reimplementing enrichment.

Run the two tails separately. Factor sign is arbitrary unless anchored
(:func:`cryptex.interpret.loadings.anchor_signs`), so an unsigned or
mis-oriented single list will conflate "genes this factor raises" with "genes it
lowers".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loadings import top_genes


def write_ranked_lists(
    loadings: pd.DataFrame,
    out_dir: str | Path,
    n: int = 200,
    tails: tuple[str, ...] = ("positive", "negative"),
) -> list[Path]:
    """Write one ``<factor>_<tail>.txt`` of gene symbols per factor and tail."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for factor in loadings.index:
        for tail in tails:
            genes = top_genes(loadings, factor, n=n, tail=tail).index
            path = out_dir / f"{factor}_{tail}.txt"
            path.write_text("\n".join(map(str, genes)) + "\n")
            written.append(path)
    return written


def write_loading_matrix(loadings: pd.DataFrame, path: str | Path) -> Path:
    """Write the full [n_factors, n_genes] loading matrix as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(path)
    return path
