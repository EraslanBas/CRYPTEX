# CRYPTEX

**Identifiable discovery of gene programs from perturbation-by-context screens.**

**Documentation: https://eraslanbas.github.io/CRYPTEX/**

## What it is for

Perturbation screens are increasingly run in more than one context: the same
CRISPR library under a panel of drugs, across cell lines or genotypes, over a
time course, or in cells from different donors. The result is a three-way array,
perturbation × context × gene, in which every entry of a response vector says
how one perturbation moved one gene in one context.

Each of those response vectors is a messy, high-dimensional readout. It spans
thousands of genes, it carries estimation noise from a finite number of cells,
and underneath it is a mixture of a small number of gene programs that the
perturbation switched on or off. CRYPTEX recovers that hidden structure:

- a **shared set of gene programs** (which genes move together), the same for
  every perturbation and every context;
- for every perturbation in every context, **how strongly it engages each
  program**;
- and from those, which part of a perturbation's effect is the **same
  everywhere**, which part is the **context's own**, and which part is an
  **interaction**: where a perturbation's role is rewired by the context it acts
  in.

## Why it matters

**The context dependence is the point of the experiment.** A multi-context
screen is run to find out how a gene's function changes under a drug, in another
genetic background or at another time. Analysing each context separately gives
programs that cannot be compared across contexts; pooling all responses into one
matrix forgets which perturbation and which context produced each row. CRYPTEX
keeps the design: one program space for the whole screen, so a perturbation's
effect in one context and in another are read on the same axes, and their
difference is the interaction.

**Programs should be reproducible, not an artefact of the run.** Standard
factorisations of a response matrix (PCA, and the many methods built on it)
pin down the subspace the programs live in but not the individual axes inside
it: any rotation of the factors fits the data equally well, so the "programs"
you read off are one arbitrary choice among many. CRYPTEX uses the
perturbation and context labels as extra information. A discriminator that has
to tell whether a response belongs to its (perturbation, context) label, and is
only allowed to score each program separately, can succeed only when the axes
are the underlying programs rather than mixtures of them. That is what makes the
programs identifiable, and the evidence for it is that they recur across
independent fits.

**The output is directly interpretable.** There is exactly one map between
programs and genes, a loading matrix in log-fold-change units, so a program's
gene list is read straight off it, and each program's activity has a clear sign:
up, down, or untouched relative to unperturbed cells of the same context.

**It is tested on combinations it has never seen.** Models are evaluated on
held-out (perturbation, context) pairs whose perturbation and context were each
seen in training but never together. A model that memorised the training pairs
scores at chance there; one that learned the programs, and how contexts reweight
them, does not.

## How it works

The programs are `B = A V` (programs × genes). CRYPTEX separates *which
subspace* they span from *which axes* inside it are the programs:

1. **The subspace.** `V` (`d` × genes) is the PCA of the training pairs' mean
   responses (each pair's full-data row and disjoint pseudobulks averaged;
   uncentred, since zero means "no response"). It is computed once and fixed;
   only the `d` × `d` matrix `A` is learned.
2. **The starting axes.** `A` starts at the joint diagonalisation of the
   per-context covariances of the pair means inside `V`: the axes along which
   the contexts' differences are cleanest.
3. **The axes, refined.** A lasso fits every response through `B` with as few
   programs as it can, each row on its own, so axes on which perturbations act
   sparsely are cheaper. A discriminator must tell
   whether a response belongs to its (perturbation, context) label and may score
   each program only separately, which is what makes the axes identifiable. A
   prior on the programs rewards few genes per program and forbids two programs
   from pointing in nearly the same direction.

The activities are the lasso fit, so there is one map between programs and
genes. The loss is

    L = L_disc + alpha * L_lasso + gamma * L_genes + mu * L_distinct

```python
from cryptex.train import TrainConfig, fit
model, history = fit(X, perturbation_idx, context_idx, target_col,
                     config=TrainConfig(), train_rows=train, val_rows=val)
```

```bash
python scripts/fit_model.py ...        # one model
python scripts/sweep_grid.py ...       # a grid of models, several seeds
python scripts/read_test.py ...        # the single test read
```

The defaults are the model's structure; `d`, `lambda`, `alpha`, `gamma` and the
label-network sizes are dataset-specific and tuned on validation. See the
documentation for every choice and the reasoning behind it.

## Layout

```
src/cryptex/      the method
    de/                 shrunken-logFC pipeline (Welch + ashr), per context
    data/               gene filter, on-target mask, leak-free splits, the subspace V
    models/             loadings B = A V, lasso activities, per-component head, label network
    objectives/         label-swap discrimination, masked reconstruction error
    interpret/          loadings, sign anchoring, effect decomposition, enrichment export
    evaluation/         span test, cross-seed stability, held-out evaluation
    train.py            training loop and evaluation
src/perturbseq_sim/     perturb-seq simulator
src/baselines/          PCA, linear ICA, joint diagonalisation, adapters
scripts/                CLIs: logFC matrices, split, fit, sweep, summaries, test read
docs/source/            documentation site (Sphinx)
notebooks/              simulator and analysis notebooks
tests/                  unit tests, no screen data needed
```

The distribution is `cryptex-screens`; the import name is **`cryptex`**.

## Install

```bash
pip install -e .                 # or run with PYTHONPATH=src
python -m pytest -q tests/
```

Building logFC matrices from single-cell data additionally needs R with `ashr`.

## Data

The repository holds code only. Data artifacts (matrices, splits, sweep
outputs) live under `$CRYPTEX_ROOT`; resolve paths with
`cryptex.paths`.

## Documentation

```bash
pip install -r docs/requirements.txt
make -C docs html                # -> docs/_build/html/index.html
```

The site is rebuilt and published to GitHub Pages on every push to `main`
(`.github/workflows/docs.yml`).
