"""Pseudo-replicates, gene filtering and label augmentation."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest




def test_affected_counts_uses_main_rows_only():
    """Subsample rows are re-estimates of the same pair, so counting them would
    multiply every gene's count without adding information."""
    import numpy as np
    import pandas as pd
    from cryptex.data import affected_counts

    X = pd.DataFrame(
        [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.0, 0.4]],
        columns=["A", "B"],
    )
    meta = pd.DataFrame({"variant": ["main", "subsample", "subsample", "main"]})
    counts = affected_counts(X, meta, threshold=0.1)
    assert counts["A"] == 1          # one main row, not three
    assert counts["B"] == 1
    all_rows = affected_counts(X, meta, threshold=0.1, variant=None)
    assert all_rows["A"] == 3


def test_select_genes_applies_the_minimum():
    import numpy as np
    import pandas as pd
    from cryptex.data import select_genes

    X = pd.DataFrame(
        [[0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.05]],
        columns=["keep", "borderline", "drop"],
    )
    meta = pd.DataFrame({"variant": ["main"] * 3})
    assert list(select_genes(X, meta, 3, threshold=0.1)) == [True, False, False]
    assert list(select_genes(X, meta, 1, threshold=0.1)) == [True, True, False]


def test_affected_counts_rejects_misaligned_meta():
    import numpy as np
    import pandas as pd
    import pytest
    from cryptex.data import affected_counts

    X = pd.DataFrame([[0.5], [0.5]], columns=["A"])
    with pytest.raises(ValueError, match="align"):
        affected_counts(X, pd.DataFrame({"variant": ["main"]}), threshold=0.1)


def test_disjoint_scheme_gives_non_overlapping_equal_size_pseudobulks():
    from cryptex.de.shards import plan_shard_rows
    p = np.array(["A"] * 350 + ["B"] * 1500 + ["C"] * 90)
    cell_index, labels, plan = plan_shard_rows(p, 10, scheme="disjoint", replicate_cells=100, seed=1)
    sub = plan[plan.variant == "subsample"]
    assert set(sub.n_cells_planned) == {100}
    assert sub.groupby("perturbation").size().to_dict() == {"A": 3, "B": 10}
    for pert in ("A", "B"):
        groups = [set(cell_index[labels == lab]) for lab in sub.label[sub.perturbation == pert]]
        assert sum(len(a & b) for i, a in enumerate(groups) for b in groups[i + 1:]) == 0
        assert all(g <= set(np.flatnonzero(p == pert)) for g in groups)
