# Ideas to experiment with later

A running list of ideas we decided to revisit. Newest at the bottom of each section. The current
model is described in `CURRENT_MODEL.md` and on the documentation site.

## The subspace V

- **Revisit the label-driven (signal-to-noise) subspace.** V from the pair rows' noise: total
  covariance of the pair means minus the within-pair noise covariance, keeping the directions with
  the most perturbation-driven variation per unit of noise (the "SNR" and RCA variants are
  implemented: `subspace_method="snr" | "rca"`). The current model uses PCA of the pair means.
  Test again with the disjoint pseudobulks (independent noise between a pair's rows), on held-out
  signal-to-noise and on stability across halves of the pairs. *(noted 2026-10-19)*
- **Free B** (programs leave the PCA subspace), started from the fixed-V solution; needed if
  programs should truly contain only ~200 genes.

## Choosing the axes / identifiability

- **Random starts per seed**: the decisive test of whether the axes are determined by the data
  rather than by the JD start. *(next, agreed)*
- **Training on halves of the perturbations.** *(done 2026-10-22 / 26; now the main criterion)*
- **Choose d by reproducibility** (cNMF-style): ICA / JD / model stability over d on halves of the
  perturbations.
- **Context-specific variances in the loss**, not only in the start: a context-variance term or a
  penalty on within-context correlations of the pair-level activities (joint diagonalisation as a
  loss term).
- **Rotation margin loss**: score each row on the current axes and on randomly rotated axes (with
  co-rotated predictions) and require the current ones to score better.
- **Track convergence of the axes**: rotation per 100 epochs; whether drift stops; compare axes at
  the selected epoch and at epoch 1000. *(done: `scripts/program_drift.py`; programs stop moving by
  ~epoch 700; diagnostic only, training always runs all epochs)*
- **Product (AMMI-type) label structure** per program: perturbation factor x context factor, retested
  with the current head and lasso.
- **Label-side sparsity instead of data-side**: least-squares z for the discriminator, sparse
  predicted effects from the label network, to avoid the AUC cost of sparse codes.
- **Drug effects sparse in programs**: the change between DMSO and a drug touches few programs.
- **Tune lambda.** *(in sweep 1: x0.5-x4 the per-row scale, 2026-10-28)*
- **The pair as one data point** (alternative to the per-row lasso): keep a shared selection per pair
  but score the whole pair under its true and a swapped label; needs equal rows per pair to keep
  cell counts out. *(noted 2026-10-25)*
- **Check the lasso solver**: 30 FISTA steps against scikit-learn's exact `Lasso` per row on saved
  models (objective, selected programs, activities). *(noted 2026-10-25)*
- **Label-only AUC check**: validation AUC with z = 0 for every row, to measure how much a model's
  AUC comes from the label term alone. *(noted 2026-10-25)*

## Gene-axis sparsity

- **L1/L2 sparseness of each program's loadings + a hinge on near-collinear programs** (cosine over
  the full loadings, tau 0.5-0.7). *(done: in the default model, gamma 0.05, tau 0.5; gamma tuned in
  sweep 1)*
- Hub genes (large-response genes in many programs): optional down-weighting in the distinctness
  term, or reporting programs by their distinctive genes.

## Head and discriminator

- **Exact normaliser** log Z(lambda) by 1-D quadrature instead of the learned c_k(p, c).
- **Sufficient statistics**: log cosh z instead of tanh z, or statistics chosen from the shape of
  the data's activities.
- **Shuffled-data negatives as a separate term**, outside the separable head.

## Data

- More pseudobulks for deep pairs (cap 20 instead of 10); 200-cell pseudobulks.
- Benchmark nested vs disjoint rows under the current model.

## Infrastructure

- Save optimiser and random-number state with the final checkpoint, to make training resumable.
- ICA / JD / PCA baselines benchmarked on the same splits and metrics.
