"""extract -- identifiable gene programs from perturbation x context screens.

The model, in one line:

    x  ~  z B ,      B = A V

* ``B`` [n_factors, n_genes] is a **global** loading matrix, shared across every
  perturbation and every context, and it is the only interpretable output. Row
  ``k`` is program ``k``'s gene loading vector, kept at unit length. ``V`` is
  the PCA of the training pairs' mean responses (uncentred, computed once and
  fixed); only the d x d matrix ``A`` is learned, starting at the
  joint-diagonalisation axes of the per-context covariances.
* ``z`` is fitted to ``x`` through ``B`` by a lasso, each row on its own
  (``models.group_lasso``), so every row is one data point. There is no
  separate encoder, so there is one factor->gene map. ``m`` masks the
  perturbation's own transcript.

The objective (``train``):

    L = L_disc + alpha * L_lasso + gamma * L_genes + mu * L_distinct

1. **L_disc** -- a binary discriminator between each row under its own label
   ``u = (perturbation, context)`` and under a swapped label, scored by a head
   restricted to a *sum over components* (``models.heads``). Additive
   separability is not preserved under mixing, so the sum-form optimum is
   reachable only in unmixed coordinates.
2. **L_lasso** -- the lasso objective (masked reconstruction error plus the
   penalty): axes on which responses need few programs are cheaper.
3. **L_genes** and **L_distinct** -- few genes per program (L1/L2 sparseness of
   each row of ``B``) and a hinge that forbids near-collinear programs.

Full specification: the documentation (``docs/source/index.md``,
https://eraslanbas.github.io/EXTRACT/).
"""

__version__ = "0.1.0.dev0"

__all__ = [
    "data",
    "models",
    "objectives",
    "interpret",
    "evaluation",
    "train",
    "paths",
]
