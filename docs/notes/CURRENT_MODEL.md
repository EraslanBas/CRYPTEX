# Current EXTRACT configuration (defaults on `main`, updated 2026-10-28)

Chosen from the 90-model comparison on the no-MT data (`sweep_jdstats_d20_noMT/`). Running
`scripts/sweep_grid.py` with no model flags reproduces it exactly.

| Part | Setting | Flag / config field |
|---|---|---|
| Data | disjoint split, full row + 100-cell pseudobulks, pairs < 100 cells removed, mitochondrial genes removed (5,504 genes); no gene centring or scaling | `--split-dir .../pair_seed0_ctxshuffle_pgenes_disjoint_min100_noMT` |
| Programs | d = 20 | `--d 20` / `n_factors` |
| Subspace | fixed V = PCA of the training pair means; B = A V, A learned | `--subspace fixed --subspace-method pca` |
| Starting axes | joint diagonalisation of the per-context covariances of the pair means | `--axes-init jd` / `axes_init` |
| Activities | a lasso per row (since 2026-10-28; earlier runs used the group lasso per pair), the same activities used by the discriminator and in the reconstruction objective; lambda tuned in the sweep (per-row scale = median \|z\| of least-squares activities on the JD axes: 0.0129 at d = 20) | `--lasso-groups row --lasso-lambda` |
| Head | four statistics (z, z^2, \|z\|, tanh z) with a learned label-only normaliser c_k(p, c) | (the only head) |
| Discriminator | binary; false examples = label swaps only, one per row, the wrong label drawn in proportion to its rows (since 2026-10-25; earlier runs drew uniformly over the labels) | `--negative-draw rows` |
| Label network | MLP on [perturbation 64, context 8] embeddings, dropout 0.2, weight decay 0.01 | `--label-dropout 0.2 --label-weight-decay 0.01` |
| Embedding starts | perturbations: target-gene responses in the training pairs; contexts: PCA of control-cell means | `--pert-embedding-init response --context-embedding-init control-means` |
| Gene axis | L1/L2 sparseness of each program's loadings (gamma = 0.05) + hinge forbidding \|cos\| > 0.5 between programs (at least 60 degrees apart; mu = 100) | `--gene-sparsity 0.05 --distinct-tau 0.5 --distinct-weight 100` |
| Loss | L_disc + alpha * L_lasso + gamma * L_genes + mu * L_distinct, alpha = 700 | `--alpha 700` |
| Training | 1000 epochs, batch 1,024, evaluated every 3 epochs, no early stopping, best epoch by validation ROC AUC; 3 seeds | `--epochs 1000` |
| Saved per model | best and final weights, config, label levels, history (every loss term), validation logits | |

Kept as options: the group lasso per pair (`--lasso-groups pair`), least-squares activities (`--coding ls`), random starting axes, the RCA and
signal-to-noise subspaces, free B, the perturbation halves. Removed from the code (in the git
history): the distance head, the plain statistics head, product/AMMI label models, shuffled-data
negatives, precision weights, the total-correlation term, early stopping, the top-k gene prior. See `group_lasso/group_lasso.pdf` for the group lasso.

Done 2026-10-22: reproducibility across halves of the perturbations (`--perturbation-half`, `--shared-subspace`), {own, shared} V x {gene prior, none}, in `sweep_halvesGS_d20_noMT/`. Open next: a hyperparameter sweep on cross-half agreement, then random starts per seed.
