"""Split first, then build every data artifact separately per partition.

The held-out test pairs are drawn **once**, from the metadata alone, before any
operation touches the values. Everything derived from values is then built
inside a partition and never across one:

* the gene list is computed from **train/val** full-data rows only, and the
  same list is applied to both partitions;
* negatives are drawn live (label swaps), by a sampler built on one
  partition's rows.

Layout written by :func:`build_split`::

    <split_dir>/
      manifest.json          seeds, fractions, gene filter, row/pair counts
      test_pairs.tsv         the held-out (perturbation, context) pairs
      gene_list.tsv          from train/val full-data rows only
      trainval/  test/
        <ctx>_real.parquet                label + genes, measured rows
        <ctx>_row_metadata.csv

Two partitions only. Validation is drawn inside ``trainval`` at fit time with
:func:`split_val_per_perturbation`; the number of subsamples ``K`` is chosen at
fit time with :func:`subsample_mask`. Both are functions of the pair and a
seed, so every model config sees the same draw.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
import re
from dataclasses import dataclass, field

SUB_RE = re.compile(r"__sub(\d+)$")
from .gene_filter import gene_list_table, load_gene_list

PARTITIONS = ("trainval", "test")


def _pair_key(meta: pd.DataFrame) -> pd.Series:
    return meta.perturbation.astype(str) + "|" + meta.context.astype(str)


def _stable_uniform(keys, seed: int) -> np.ndarray:
    """Uniform [0, 1) per key, fixed by key and seed alone (not by row order
    or by which other rows are present). ``hash()`` is salted per process."""
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        digest = hashlib.md5(f"{k}:{seed}".encode()).digest()
        out[i] = int.from_bytes(digest[:8], "little") / 2**64
    return out


# ---- the three draws --------------------------------------------------------


def draw_test_pairs(
    meta: pd.DataFrame, test_frac: float = 0.1, seed: int = 0
) -> list[tuple[str, str]]:
    """Hold out ``test_frac`` of all ``(perturbation, context)`` pairs.

    Every perturbation keeps **at least one pair** outside test: a perturbation
    with all its pairs held out has an embedding ``e_p`` that is never trained,
    so its test rows are unpredictable by construction. Pairs are visited in a
    random order and a pair is skipped if taking it would remove its
    perturbation's last remaining pair.
    """
    pairs = (
        meta[["perturbation", "context"]].astype(str).drop_duplicates()
        .sort_values(["perturbation", "context"]).to_numpy()
    )
    n_test = int(round(test_frac * len(pairs)))
    if not 0 < n_test < len(pairs):
        raise ValueError(f"test_frac={test_frac} gives {n_test} of {len(pairs)} pairs")

    remaining = pd.Series(pairs[:, 0]).value_counts().to_dict()
    chosen = []
    for i in np.random.default_rng(seed).permutation(len(pairs)):
        p = pairs[i, 0]
        if remaining[p] <= 1:
            continue
        chosen.append((pairs[i, 0], pairs[i, 1]))
        remaining[p] -= 1
        if len(chosen) == n_test:
            break
    if len(chosen) < n_test:
        raise ValueError(
            f"only {len(chosen)} pairs can be held out while keeping one pair "
            f"per perturbation; {n_test} requested"
        )
    return sorted(chosen)


def perturbation_half(perturbations, half: int, seed: int = 0) -> np.ndarray:
    """Boolean mask of the rows whose perturbation is in one random half of the
    perturbations (``half`` 0 or 1). The halves partition the perturbations:
    every pair of a perturbation, in every context, goes to the same half, so
    models trained on the two halves never see a common perturbation. The
    assignment depends only on the set of perturbation names and ``seed``."""
    if half not in (0, 1):
        raise ValueError(f"half must be 0 or 1, got {half}")
    perts = np.asarray(perturbations).astype(str)
    names = np.unique(perts)
    order = np.random.default_rng(seed).permutation(len(names))
    first = set(names[order[: len(names) // 2]])
    in_first = np.isin(perts, list(first))
    return in_first if half == 0 else ~in_first


def split_val_per_perturbation(
    meta: pd.DataFrame, val_frac: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Boolean mask over rows: ``True`` for validation pairs.

    Grouped per perturbation: each perturbation holds out about ``val_frac`` of
    its pairs and **always keeps at least one** in train, so every validation
    pair's perturbation and context were both trained and validation measures
    the interaction. ``val_frac * n_pairs`` is rounded stochastically, so
    perturbations with few pairs still contribute and the total comes out near
    ``val_frac``. All rows of a pair move together.
    """
    pairs = (
        meta[["perturbation", "context"]].astype(str).drop_duplicates()
        .sort_values(["perturbation", "context"])
    )
    rng = np.random.default_rng(seed)
    held = []
    for pert, grp in pairs.groupby("perturbation", sort=True):
        n = len(grp)
        target = val_frac * n
        k = int(np.floor(target)) + int(rng.random() < target - np.floor(target))
        k = min(k, n - 1)
        if k:
            ctxs = grp.context.to_numpy()[rng.choice(n, k, replace=False)]
            held += [f"{pert}|{c}" for c in ctxs]
    return _pair_key(meta).isin(set(held)).to_numpy()


def subsample_mask(
    meta: pd.DataFrame, n_subsamples: int, seed: int = 0
) -> np.ndarray:
    """Keep the full-data row and ``n_subsamples`` of each pair's subsamples.

    The chosen subsamples are the first ``K`` of a random ordering fixed per
    ``(perturbation, context)`` and ``seed``, so the sets are **nested**
    (K=1 inside K=2 inside ... K=10) and a difference between two K values is
    due to K, not to which subsamples were drawn. A synthetic row has the same
    pair and label as its real twin, so the two are kept or dropped together.
    A pair with fewer than ``K`` subsamples keeps all it has.
    """
    labels = meta.label.astype(str)
    sub = labels.str.extract(SUB_RE.pattern, expand=False)
    is_sub = sub.notna().to_numpy()
    keep = ~is_sub
    if n_subsamples <= 0 or not is_sub.any():
        return keep
    frame = pd.DataFrame({
        "pair": _pair_key(meta).to_numpy()[is_sub],
        "label": labels.to_numpy()[is_sub],
    })
    # rank distinct labels within a pair (a synthetic twin shares the label)
    uniq = frame.drop_duplicates()
    uniq = uniq.assign(u=_stable_uniform((uniq.pair + "|" + uniq.label).to_numpy(), seed))
    uniq["rank"] = uniq.groupby("pair").u.rank(method="first") - 1
    rank = frame.merge(uniq, on=["pair", "label"], how="left")["rank"].to_numpy()
    keep[np.nonzero(is_sub)[0]] = rank < n_subsamples
    return keep


# ---- building and loading ---------------------------------------------------


def _write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_split(
    matrices_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    test_frac: float = 0.1,
    split_seed: int = 0,
    min_affected: int = 151,
    threshold: float = float(np.log(1.2)),
    contexts: list[str] | None = None,
    keep_perturbed_genes: bool = False,
    control_label: str = "non-targeting",
    gene_list=None,
    min_pair_cells: int = 0,
) -> dict:
    """Draw the test pairs, then write both partitions. Returns the manifest.

    ``keep_perturbed_genes`` also keeps every measured gene that is the target
    of some perturbation, whether or not it passes the affected-count filter,
    so that each targeted gene's own response is in the matrix.

    Two passes over the matrices: the first counts affected genes on train/val
    full-data rows, the second writes each context's rows, gene-filtered, into
    its partition.

    Refuses to write into a directory that already holds a manifest: a split
    is fixed once made, and every fit that used it depends on that.
    """
    from .gene_filter import affected_counts
    import pyarrow.parquet as pq

    matrices_dir = Path(matrices_dir or paths.matrices())
    out_dir = Path(out_dir or paths.root() / "splits" / f"pair_seed{split_seed}")
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} already holds a split; use a new directory")
    for part in PARTITIONS:
        (out_dir / part).mkdir(parents=True, exist_ok=True)
    started = time.time()

    found = sorted(matrices_dir.glob("*_PosteriorMean.parquet"))
    names = [f.name.replace("_PosteriorMean.parquet", "") for f in found]
    if contexts:
        names = [n for n in names if n in set(contexts)]
    # `_row`: each row's position in its matrix file, kept through any filtering
    metas = {n: pd.read_csv(matrices_dir / f"{n}_row_metadata.csv").assign(
        _row=lambda d: np.arange(len(d))) for n in names}
    meta_all = pd.concat(metas.values(), ignore_index=True)

    # ---- 1. the split, from metadata alone --------------------------------
    test_pairs = draw_test_pairs(meta_all, test_frac=test_frac, seed=split_seed)
    pd.DataFrame(test_pairs, columns=["perturbation", "context"]).to_csv(
        out_dir / "test_pairs.tsv", sep="\t", index=False)
    held = {f"{p}|{c}" for p, c in test_pairs}
    print(f"[split] {len(test_pairs):,} test pairs of "
          f"{meta_all.groupby(['perturbation', 'context']).ngroups:,}", flush=True)
    # ---- 1b. drop unreliable pairs (few cells), AFTER the draw, so the test
    # set stays a subset of the same held-out pairs --------------------------
    dropped = set()
    if min_pair_cells:
        main_rows = meta_all[meta_all.variant == "main"]
        low = main_rows.n_cells < min_pair_cells
        dropped = set(_pair_key(main_rows[low]))
        kept_tv = main_rows[~low & ~_pair_key(main_rows).isin(held)]
        trained = set(kept_tv.perturbation.astype(str))
        orphan = {k for k in held if k.split("|")[0] not in trained}
        dropped |= orphan
        held -= dropped
        test_pairs = [(p, c) for p, c in test_pairs if f"{p}|{c}" in held]
        pd.DataFrame(test_pairs, columns=["perturbation", "context"]).to_csv(
            out_dir / "test_pairs.tsv", sep="\t", index=False)
        pd.DataFrame(sorted(dropped), columns=["pair"]).to_csv(out_dir / "dropped_pairs.tsv",
                                                               sep="\t", index=False)
        metas = {n: mm[~_pair_key(mm).isin(dropped)].reset_index(drop=True) for n, mm in metas.items()}
        meta_all = pd.concat(metas.values(), ignore_index=True)
        print(f"[filter] dropped {int(low.sum()):,} pairs with < {min_pair_cells} cells and "
              f"{len(orphan):,} test pairs whose perturbation had no training pair left; "
              f"{len(test_pairs):,} test pairs remain", flush=True)

    # ---- 2. gene list from train/val full-data rows -----------------------
    counts, genes_all = None, None
    for n in names:
        m = metas[n]
        rows = m["_row"].to_numpy()[((m.variant == "main") & ~_pair_key(m).isin(held)).to_numpy()]
        table = pq.read_table(matrices_dir / f"{n}_PosteriorMean.parquet").take(rows)
        genes = [c for c in table.column_names if c != "label"]
        if genes_all is None:
            genes_all = genes
        elif genes != genes_all:
            raise ValueError(f"{n}: gene columns differ from {names[0]}")
        values = table.select(genes).to_pandas().to_numpy(dtype=np.float32)
        c = affected_counts(values, m.set_index("_row").loc[rows].reset_index(), threshold=threshold, variant="main")
        counts = c.to_numpy() if counts is None else counts + c.to_numpy()
        del table, values
    counts = pd.Series(counts, index=genes_all, name="n_affected")
    tv_main = meta_all[(meta_all.variant == "main") & ~_pair_key(meta_all).isin(held)]
    gl = gene_list_table(counts, tv_main, min_affected, threshold,
                         out_dir / "gene_list.tsv")
    if gene_list is not None:
        # keep exactly these genes (e.g. another split's list, so that two
        # matrices sets are compared on the same genes)
        forced = set(map(str, gene_list))
        missing = forced - set(gl.gene.astype(str))
        if missing:
            raise ValueError(f"{len(missing)} genes of gene_list not in the matrices, e.g. {sorted(missing)[:3]}")
        gl["keep"] = gl.gene.astype(str).isin(forced)
        gl.to_csv(out_dir / "gene_list.tsv", sep="\t", index=False)
    n_filter = int(gl.keep.sum())
    if keep_perturbed_genes:
        targets = set(meta_all.perturbation.astype(str)) - {control_label}
        gl["perturbed"] = gl.gene.astype(str).isin(targets)
        gl["keep"] = gl.keep | gl.perturbed
        path = out_dir / "gene_list.tsv"
        header = [ln for ln in path.read_text().splitlines(True) if ln.startswith("#")]
        header.append(f"# plus perturbed genes: {int((gl.perturbed & (gl.n_affected < min_affected)).sum())} "
                      f"targets kept below the filter\n")
        with open(path, "w") as fh:
            fh.writelines(header)
            gl.to_csv(fh, sep="\t", index=False)
    genes = gl.loc[gl.keep, "gene"].tolist()
    print(f"[genes] {len(genes):,} of {len(genes_all):,} kept "
          f"(|logFC| > {threshold:.5f} in >= {min_affected} train/val pairs: {n_filter:,}"
          + (f"; plus {len(genes) - n_filter:,} perturbed genes" if keep_perturbed_genes else "")
          + ")", flush=True)

    # ---- 3. per context, per partition ------------------------------------
    per_context = []
    for n in names:
        m = metas[n].copy()
        table = pq.read_table(matrices_dir / f"{n}_PosteriorMean.parquet",
                              columns=["label"] + genes).take(m["_row"].to_numpy()).to_pandas()
        m = m.drop(columns="_row")
        if not (table.label.to_numpy() == m.label.to_numpy()).all():
            raise ValueError(f"{n}: matrix and metadata row order disagree")
        in_test = _pair_key(m).isin(held).to_numpy()
        for part, mask in (("trainval", ~in_test), ("test", in_test)):
            rows = np.nonzero(mask)[0]
            block = table.iloc[rows].reset_index(drop=True)
            _write_atomic_parquet(block, out_dir / part / f"{n}_real.parquet")
            m.iloc[rows].assign(partition=part).to_csv(
                out_dir / part / f"{n}_row_metadata.csv", index=False)
            per_context.append({"context": n, "partition": part, "rows": len(rows),
                                "pairs": int(_pair_key(m.iloc[rows]).nunique())})
            del block
        print(f"[write] {n}: {int((~in_test).sum()):,} trainval / "
              f"{int(in_test.sum()):,} test rows", flush=True)
        del table

    pc = pd.DataFrame(per_context)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_matrices": str(matrices_dir),
        "split_level": "pair",
        "test_frac": test_frac,
        "split_seed": split_seed,
        "rule": "every perturbation keeps >= 1 pair in trainval",
        "gene_filter": {"threshold": threshold, "min_affected": min_affected,
                        "computed_on": "trainval full-data rows",
                        "n_genes": len(genes), "n_genes_before": len(genes_all),
                        "n_passing_filter": n_filter,
                        "keep_perturbed_genes": keep_perturbed_genes},
        "contexts": names,
        "partitions": {
            part: {"rows": int(pc[pc.partition == part].rows.sum()),
                   "pairs": int(pc[pc.partition == part].pairs.sum())}
            for part in PARTITIONS
        },
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_partition(
    split_dir: str | Path,
    partition: str,
    contexts: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """``(X, meta, genes)`` for one partition: the measured rows of every
    context, in file order. (Older splits may also hold gene-shuffled copies of
    the rows; they are not read.)"""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    names = manifest["contexts"]
    if contexts:
        missing = set(contexts) - set(names)
        if missing:
            raise FileNotFoundError(f"not in this split: {sorted(missing)}")
        names = [n for n in names if n in set(contexts)]
    genes = np.asarray(load_gene_list(split_dir / "gene_list.tsv"))
    cols = [str(g) for g in genes]

    real_x, real_m = [], []
    for n in names:
        d = split_dir / partition
        m = pd.read_csv(d / f"{n}_row_metadata.csv")
        block = pd.read_parquet(d / f"{n}_real.parquet")
        if not (block.label.to_numpy() == m.label.to_numpy()).all():
            raise ValueError(f"{partition}/{n}: matrix and metadata disagree")
        real_x.append(block[cols].to_numpy(dtype=np.float32))
        real_m.append(m)
        del block

    X = np.vstack(real_x)
    meta = pd.concat(real_m, ignore_index=True)
    return X, meta, genes


# ---- pair-level splits of an in-memory table (used by tests and small fits) ----

@dataclass
class Split:
    """Boolean masks over rows, plus what was held out.

    ``val`` is ``None`` for a two-way split. Prefer three ways: the validation
    half is what you are allowed to look at repeatedly -- early stopping,
    ``alpha``, ``d`` -- and the test half is what you look at once, at the end.
    Choosing the stopping epoch on the test half turns it into a validation set
    and the number reported from it is no longer held out.
    """

    train: np.ndarray
    test: np.ndarray
    level: str
    held_out: list
    val: np.ndarray | None = None
    held_out_val: list = field(default_factory=list)

    @property
    def three_way(self) -> bool:
        return self.val is not None

    def summary(self, meta: pd.DataFrame) -> dict:
        out = {
            "level": self.level,
            "train_rows": int(self.train.sum()),
            "test_rows": int(self.test.sum()),
            "held_out": len(self.held_out),
            "train_perts": int(meta[self.train].perturbation.nunique()),
            "test_perts": int(meta[self.test].perturbation.nunique()),
            "train_contexts": int(meta[self.train].context.nunique()),
            "test_contexts": int(meta[self.test].context.nunique()),
        }
        if self.val is not None:
            out.update(
                val_rows=int(self.val.sum()),
                held_out_val=len(self.held_out_val),
                val_perts=int(meta[self.val].perturbation.nunique()),
                val_contexts=int(meta[self.val].context.nunique()),
            )
        return out


def make_split(
    meta: pd.DataFrame,
    level: str = "pair",
    test_frac: float = 0.1,
    seed: int = 0,
    held_out_contexts: list[str] | None = None,
    val_frac: float = 0.0,
    protect_single_context: bool = True,
) -> Split:
    """Leak-free split at the requested granularity.

    ``level``
        ``"pair"`` holds out ``(perturbation, context)`` combinations -- both
        sides seen elsewhere, so the model must supply the interaction.
        ``"perturbation"`` holds out whole perturbations across every context.
        ``"context"`` holds out whole contexts (pass ``held_out_contexts``).

    ``val_frac``
        Non-zero makes it a three-way split, drawing the validation units
        disjointly from the test units. Use validation for early stopping and
        hyperparameters and keep test for a single final read; otherwise the
        stopping epoch is chosen on the test set and its numbers are no longer
        held out.

    ``protect_single_context``
        At ``level="pair"``, keep every pair of a perturbation observed in only
        ONE context out of the holdout draw, so it always trains. Such a
        perturbation has a single pair; if that pair is held out, ``e_p`` is
        never trained and the row is unpredictable by construction -- a noise
        floor on the held-out score rather than a test of anything. On the
        12-context screen this affected 3 perturbations at seed 0. Fractions
        stay relative to *all* pairs, so the held-out count is unchanged; only
        the pool drawn from shrinks.

    In every case all 11 samples of an affected ``(perturbation, context)`` move
    together, which is checked before returning.
    """
    rng = np.random.default_rng(seed)
    pert = meta.perturbation.to_numpy().astype(str)
    ctx = meta.context.to_numpy().astype(str)
    val = held_val = None

    if level == "pair":
        pairs = pd.MultiIndex.from_arrays([pert, ctx])
        unique = pairs.unique()
        n_test = max(1, int(round(test_frac * len(unique))))
        n_val = max(1, int(round(val_frac * len(unique)))) if val_frac else 0
        if n_test + n_val > len(unique):
            raise ValueError(
                f"test_frac + val_frac would hold out {n_test + n_val} of "
                f"{len(unique)} pairs"
            )

        eligible = np.arange(len(unique))
        if protect_single_context:
            # A perturbation with one pair has nothing left to train e_p on if
            # that pair is held out. Keep those pairs in train.
            n_by_pert = pd.Series(unique.get_level_values(0)).value_counts()
            singles = set(n_by_pert.index[n_by_pert == 1])
            if singles:
                keep_in_train = np.array(
                    [p in singles for p in unique.get_level_values(0)]
                )
                eligible = np.nonzero(~keep_in_train)[0]
                if n_test + n_val > len(eligible):
                    raise ValueError(
                        f"after protecting {len(singles)} single-context "
                        f"perturbations only {len(eligible)} pairs remain "
                        f"drawable, need {n_test + n_val}"
                    )
        drawn = eligible[rng.choice(len(eligible), n_test + n_val, replace=False)]
        chosen = set(unique[drawn[:n_test]])
        test = np.array([pc in chosen for pc in pairs])
        held = sorted(chosen)
        if n_val:
            chosen_val = set(unique[drawn[n_test:]])
            val = np.array([pc in chosen_val for pc in pairs])
            held_val = sorted(chosen_val)
    elif level == "perturbation":
        unique = np.unique(pert)
        n_test = max(1, int(round(test_frac * len(unique))))
        n_val = max(1, int(round(val_frac * len(unique)))) if val_frac else 0
        if n_test + n_val > len(unique):
            raise ValueError(
                f"test_frac + val_frac would hold out {n_test + n_val} of "
                f"{len(unique)} perturbations"
            )
        drawn = rng.choice(unique, n_test + n_val, replace=False)
        chosen = set(drawn[:n_test])
        test = np.isin(pert, list(chosen))
        held = sorted(chosen)
        if n_val:
            chosen_val = set(drawn[n_test:])
            val = np.isin(pert, list(chosen_val))
            held_val = sorted(chosen_val)
    elif level == "context":
        if not held_out_contexts:
            raise ValueError("level='context' needs held_out_contexts")
        missing = set(held_out_contexts) - set(np.unique(ctx))
        if missing:
            raise ValueError(f"context(s) not present: {sorted(missing)}")
        test = np.isin(ctx, held_out_contexts)
        held = sorted(held_out_contexts)
        if val_frac:
            raise ValueError(
                "level='context' does not draw a validation set; name the "
                "validation contexts explicitly with a second call"
            )
    else:
        raise ValueError(f"level must be pair, perturbation or context, got {level!r}")

    train = ~test if val is None else ~(test | val)
    split = Split(
        train=train, test=test, level=level, held_out=held,
        val=val, held_out_val=held_val or [],
    )
    _assert_no_pair_leak(meta, split)
    return split


def _assert_no_pair_leak(meta: pd.DataFrame, split: Split) -> None:
    """No (perturbation, context) may appear in more than one half."""
    key = meta.perturbation.astype(str) + "|" + meta.context.astype(str)
    halves = {"train": set(key[split.train]), "test": set(key[split.test])}
    if split.val is not None:
        halves["val"] = set(key[split.val])

    names = list(halves)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = halves[a] & halves[b]
            if both:
                raise AssertionError(
                    f"{len(both)} (perturbation, context) pairs appear in BOTH "
                    f"{a} and {b}, e.g. {sorted(both)[:3]} -- a sample-level leak"
                )

    covered = sum(m.sum() for m in
                  [split.train, split.test] + ([split.val] if split.val is not None else []))
    if int(covered) != len(meta):
        raise AssertionError(
            f"the halves cover {covered} rows but meta has {len(meta)}"
        )


def apply_split(split: Split, meta: pd.DataFrame) -> Split:
    """Re-derive a split's masks for a different row set.

    A synthetic row inherits the ``(perturbation, context)`` of the real row it
    was generated from, so it must land in the **same half** as that pair. If it
    does not, a held-out pair's synthetic twin appears in training and the
    held-out numbers stop meaning anything -- the same class of leak as a
    negative that names a held-out pair, and one :func:`make_split` cannot
    catch, because it splits on ``(perturbation, context)`` and knows nothing
    about ``is_real``.

    Rather than re-drawing (which would put the twins in different halves),
    this re-applies the *units* the original split held out.
    """
    pert = meta.perturbation.astype(str).to_numpy()
    ctx = meta.context.astype(str).to_numpy()
    val = None

    if split.level == "pair":
        held = {(str(a), str(b)) for a, b in split.held_out}
        held_v = {(str(a), str(b)) for a, b in split.held_out_val}
        key = list(zip(pert, ctx))
        test = np.array([k in held for k in key])
        if held_v:
            val = np.array([k in held_v for k in key])
    elif split.level == "perturbation":
        test = np.isin(pert, [str(x) for x in split.held_out])
        if split.held_out_val:
            val = np.isin(pert, [str(x) for x in split.held_out_val])
    elif split.level == "context":
        test = np.isin(ctx, [str(x) for x in split.held_out])
    else:  # pragma: no cover - make_split validates this
        raise ValueError(f"unknown split level {split.level!r}")

    train = ~test if val is None else ~(test | val)
    out = Split(train=train, test=test, level=split.level,
                held_out=split.held_out, val=val,
                held_out_val=split.held_out_val)
    _assert_no_pair_leak(meta, out)
    return out
