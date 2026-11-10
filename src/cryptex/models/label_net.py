"""(perturbation, context) -> per-component coefficients theta (and c_k).

Deliberately expressive. The identifiability constraint in
``models.heads.PerComponentHead`` restricts only *how* theta meets the
features -- never how theta itself is computed. All of the perturbation x
context interaction lives here and is unconstrained.

Factorised rather than one free embedding row per observed ``(p, c)`` tuple.
That distinction decides whether the model learns or memorises: a free lookup
table has one row per pair, so with a single observation per pair the optimal
discriminator is ``answer real iff x == x_u`` -- a hash table, reachable
without any notion of components. Sharing ``e_p`` across contexts and ``e_c``
across perturbations drops the parameter count from ``n_pairs * d`` to
``(n_perturbations + n_contexts) * d`` and makes memorisation impossible.
"""

from __future__ import annotations

import torch
from torch import nn


class FactorizedLabelNet(nn.Module):
    """Map ``(perturbation_idx, context_idx)`` to the head's coefficients
    [d, n_basis] (``n_basis`` = the head's outputs per factor: theta and c_k).

    Parameters
    ----------
    embedding_dim
        Width of ``e_p`` and ``e_c``, unless ``context_embedding_dim`` is given.
    context_embedding_dim
        Width of ``e_c`` when it differs from ``e_p``'s.
    hidden
        Width of the interaction MLP. Set ``hidden=0`` for a purely bilinear
        (additive-in-embeddings) map, which is the low-rank interaction model.
    """

    def __init__(
        self,
        n_perturbations: int,
        n_contexts: int,
        n_factors: int,
        n_basis: int,
        embedding_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.0,
        context_embedding_dim: int | None = None,
    ):
        super().__init__()
        ctx_dim = embedding_dim if context_embedding_dim is None else context_embedding_dim
        self.n_factors = n_factors
        self.n_basis = n_basis

        self.perturbation_embedding = nn.Embedding(n_perturbations, embedding_dim)
        self.context_embedding = nn.Embedding(n_contexts, ctx_dim)
        nn.init.normal_(self.perturbation_embedding.weight, std=0.02)
        nn.init.normal_(self.context_embedding.weight, std=0.02)

        out_dim = n_factors * n_basis
        # dropout on the concatenated embeddings and the hidden layer:
        # regularises against memorising the training pairs
        if hidden:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim + ctx_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim + ctx_dim, out_dim))

    def forward(
        self, perturbation_idx: torch.Tensor, context_idx: torch.Tensor
    ) -> torch.Tensor:
        e_p = self.perturbation_embedding(perturbation_idx)
        e_c = self.context_embedding(context_idx)
        lam = self.head(torch.cat([e_p, e_c], dim=-1))
        return lam.view(-1, self.n_factors, self.n_basis)
