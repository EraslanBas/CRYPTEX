"""Split-first data layout: draws, nesting, and partition isolation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptex.data.splits import (
    build_split,
    draw_test_pairs,
    load_partition,
    split_val_per_perturbation,
    subsample_mask,
)


def _meta(n_pert=30, n_ctx=5, n_sub=10, seed=0, ragged=True):
    """Rows like the real metadata: a full row plus n_sub subsamples per pair.
    With ``ragged`` some perturbations are in few contexts, some have few
    subsamples."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pert):
        p = f"P{i:02d}"
        ctxs = [f"C{j}" for j in range(n_ctx)]
        if ragged and i % 7 == 0:
            ctxs = ctxs[:1]                          # single-context perturbation
        elif ragged and i % 5 == 0:
            ctxs = ctxs[:2]
        for c in ctxs:
            k = n_sub if not (ragged and i % 4 == 0) else int(rng.integers(0, 4))
            rows.append({"perturbation": p, "context": c, "label": p,
                         "variant": "main", "n_cells": 500})
            for s in range(k):
                rows.append({"perturbation": p, "context": c,
                             "label": f"{p}__sub{s:02d}", "variant": "subsample",
                             "n_cells": 50 + 40 * s})
    m = pd.DataFrame(rows)
    m.insert(0, "row_id", m.context + "|" + m.label)
    return m


def _pairs(m):
    return set(zip(m.perturbation, m.context))


def test_test_pairs_leave_every_perturbation_a_trainval_pair():
    m = _meta()
    for seed in range(5):
        test = set(draw_test_pairs(m, test_frac=0.3, seed=seed))
        left = _pairs(m) - test
        assert {p for p, _ in left} == set(m.perturbation)
        n = len(_pairs(m))
        assert len(test) == round(0.3 * n)


def test_val_split_keeps_a_train_pair_per_perturbation_and_hits_the_fraction():
    m = _meta(n_pert=200, n_ctx=8, ragged=True)
    val = split_val_per_perturbation(m, val_frac=0.1, seed=0)
    train_pairs = _pairs(m[~val])
    val_pairs = _pairs(m[val])
    assert not train_pairs & val_pairs
    assert {p for p, _ in val_pairs} <= {p for p, _ in train_pairs}
    # all rows of a pair move together
    key = m.perturbation + "|" + m.context
    assert (pd.Series(val).groupby(key.to_numpy()).nunique() == 1).all()
    frac = len(val_pairs) / len(_pairs(m))
    assert 0.07 < frac < 0.13, frac


def test_subsample_mask_is_nested_and_keeps_full_rows():
    m = _meta(ragged=True)
    masks = {k: subsample_mask(m, k, seed=3) for k in (1, 2, 3, 5, 7, 10)}
    full = (m.variant == "main").to_numpy()
    for k, mk in masks.items():
        assert mk[full].all(), "full-data rows are always kept"
        per_pair = (pd.Series(mk & ~full)
                    .groupby((m.perturbation + "|" + m.context).to_numpy()).sum())
        avail = (pd.Series(~full)
                 .groupby((m.perturbation + "|" + m.context).to_numpy()).sum())
        assert (per_pair == np.minimum(avail, k)).all(), k
    ks = sorted(masks)
    for a, b in zip(ks, ks[1:]):
        assert not (masks[a] & ~masks[b]).any(), f"K={a} not inside K={b}"


def test_subsample_mask_is_a_function_of_the_pair_not_the_row_set():
    """Same pair, same seed, same pick, whatever other rows are present --
    so train/val and test (or real and synthetic) agree."""
    m = _meta(ragged=False)
    full = subsample_mask(m, 3, seed=1)
    part = m[m.context == "C2"]
    assert (subsample_mask(part, 3, seed=1) == full[part.index]).all()
    twins = pd.concat([m, m], ignore_index=True)
    both = subsample_mask(twins, 3, seed=1)
    assert (both[: len(m)] == both[len(m):]).all()


def _write_matrices(d, meta, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{j}" for j in range(n_genes)]
    for ctx, m in meta.groupby("context", sort=True):
        m = m.reset_index(drop=True)
        vals = rng.normal(scale=0.3, size=(len(m), n_genes)).astype(np.float32)
        vals[:, 0] *= 0.01                           # a gene nothing moves
        frame = pd.DataFrame(vals, columns=genes)
        frame.insert(0, "label", m.label.to_numpy())
        frame.to_parquet(d / f"{ctx}_PosteriorMean.parquet", index=False)
        m.to_csv(d / f"{ctx}_row_metadata.csv", index=False)
    return genes


def test_build_split_isolates_the_partitions(tmp_path):
    meta = _meta(n_pert=40, n_ctx=4, ragged=True)
    mats = tmp_path / "matrices"; mats.mkdir()
    genes = _write_matrices(mats, meta)
    out = tmp_path / "split"
    man = build_split(mats, out, test_frac=0.2, split_seed=0,
                      min_affected=5, threshold=0.2)

    test_pairs = set(map(tuple, pd.read_csv(out / "test_pairs.tsv", sep="\t")
                         .astype(str).to_numpy()))
    Xtv, mtv, g = load_partition(out, "trainval")
    Xte, mte, g2 = load_partition(out, "test")

    assert list(g) == list(g2) and "G0" not in set(g)
    assert set(zip(mtv.perturbation, mtv.context)).isdisjoint(test_pairs)
    assert set(zip(mte.perturbation, mte.context)) == test_pairs
    assert len(mtv) + len(mte) == len(meta) and len(Xtv) == len(mtv)
    assert man["partitions"]["test"]["pairs"] == len(test_pairs)
    assert not list(out.glob("*/*shuffled*"))           # only measured rows are written

    # the gene list was decided on train/val full-data rows only
    header = (out / "gene_list.tsv").read_text().splitlines()[1]
    n_tv_main = int(((meta.variant == "main")
                     & ~pd.Series(list(zip(meta.perturbation, meta.context)))
                     .isin(test_pairs)).sum())
    assert f"n_pairs={n_tv_main}" in header

    with pytest.raises(FileExistsError):
        build_split(mats, out, test_frac=0.2)
    assert json.loads((out / "manifest.json").read_text())["split_seed"] == 0


def test_fit_model_and_the_single_test_read(tmp_path):
    """End to end: build a split, fit one model on trainval, read test once."""
    import subprocess
    import sys
    from pathlib import Path

    meta = _meta(n_pert=40, n_ctx=4, ragged=False)
    mats = tmp_path / "matrices"; mats.mkdir()
    _write_matrices(mats, meta, n_genes=16)
    split = tmp_path / "split"
    build_split(mats, split, test_frac=0.15, min_affected=5, threshold=0.2)

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    out = tmp_path / "models"
    proc = subprocess.run(
        [sys.executable, str(scripts / "fit_model.py"), "--split-dir", str(split),
         "--out-dir", str(out), "-d", "3", "-K", "3", "--epochs", "3", "--batch-size", "64",
         "--eval-every", "1", "--context-embedding-init", "random"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    m = pd.read_csv(out / "sweep_metrics.csv")
    assert len(m) == 1 and 0.0 <= m.val_auc.iloc[0] <= 1.0
    tag = m.tag.iloc[0]
    for f in ("model_best_", "model_final_", "config_", "history_", "levels_", "val_scores_"):
        assert list(out.glob(f"{f}{tag}.*")), f
    assert not any("test_" in c for c in m.columns)        # the sweep never reads test

    read = [sys.executable, str(scripts / "read_test.py"), str(out), "--tag", tag, "--reason", "unit test"]
    proc = subprocess.run(read, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "test_auc" in proc.stdout
    assert subprocess.run(read, capture_output=True, text=True).returncode != 0   # a repeat is refused


def test_perturbation_halves_partition_the_perturbations():
    from cryptex.data.splits import perturbation_half
    perts = np.repeat([f"G{i}" for i in range(41)], 3)          # 3 contexts per perturbation
    h0, h1 = perturbation_half(perts, 0, seed=1), perturbation_half(perts, 1, seed=1)
    assert (h0 ^ h1).all()                                       # every row in exactly one half
    assert not set(perts[h0]) & set(perts[h1])                   # no perturbation in both
    assert abs(len(set(perts[h0])) - len(set(perts[h1]))) <= 1
    assert (perturbation_half(perts[::-1], 0, seed=1) == h0[::-1]).all()   # depends on names, not order
    assert not (perturbation_half(perts, 0, seed=2) == h0).all()
