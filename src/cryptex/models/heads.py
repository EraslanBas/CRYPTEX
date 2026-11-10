"""The per-component head -- the identifiability constraint.

    s(x, u)  =  b  +  sum_k [ sum_j theta_{k,j}(u) * q_j(z_k)  -  c_k(u) ]

The sum runs over components and **nothing multiplies z_k by z_j for k != j**.
That restriction is the whole mechanism:

The negatives are label swaps (the observed row under another label), so the
optimal discriminator is the density ratio ``log p(x, u) / (p(x) p(u))``. If
the components are independent **given the label** (no marginal independence
is assumed) and each has an exponential-family distribution in the statistics
``q``, that ratio is

    sum_k [ theta_k(u) . q(z_k) - log Z_k(u) ]   + terms free of u,

i.e. additively separable in the true components, with ``c_k = log Z_k``.
Additive separability is not preserved under mixing, so a head restricted to
separable functions can reach the optimum only if its coordinates are the
components. With an unrestricted head (an MLP over the whole feature vector)
the substitution ``z -> M z`` can be absorbed by the head's first layer, the
loss cannot see ``M``, and the axes float.

One caveat on the basis: a purely quadratic head identifies nothing, because
``sum_k z_k^2`` is rotation invariant -- the same reason Gaussian sources are
unidentifiable in linear ICA. Keep at least one non-quadratic statistic
(``abs`` or ``tanh``) in the basis.

This is generalised contrastive learning (Hyvarinen, Sasaki & Turner 2019; the
exponential family of time-contrastive learning) with ``q_j`` as the
sufficient statistics and ``theta(u)`` as the label-dependent natural
parameters.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

#: Scalar statistics applied coordinate-wise to ``z``.
#: ``linear`` + ``square`` alone detect per-component mean and variance shifts
#: (a Gaussian family, and rotation-invariant in the square term); ``abs`` and
#: ``tanh`` add sensitivity to non-Gaussian shape, which is extra identifying
#: information.
BASIS_FUNCTIONS = {
    "linear": lambda z: z,
    "square": lambda z: z**2,
    "abs": torch.abs,
    "tanh": torch.tanh,
}

DEFAULT_BASIS = ("linear", "square", "abs", "tanh")


class PerComponentHead(nn.Module):
    """Score a row under a label (true or swapped), separably in the components.

    Parameters
    ----------
    basis
        Names from :data:`BASIS_FUNCTIONS`. Must include at least one
        non-quadratic statistic or the head is rotation invariant.
    normaliser
        Also take a label-only term ``c_k(u)`` per component from the label
        network and subtract it (``Cryptex`` always uses it).
    """

    def __init__(self, basis: Sequence[str] = DEFAULT_BASIS, normaliser: bool = False):
        super().__init__()
        unknown = set(basis) - set(BASIS_FUNCTIONS)
        if unknown:
            raise ValueError(f"unknown basis functions: {sorted(unknown)}")
        if set(basis) <= {"square"}:
            raise ValueError(
                "a purely quadratic basis is rotation invariant and identifies "
                "nothing; include 'abs' or 'tanh'"
            )
        self.basis = tuple(basis)
        self.bias = nn.Parameter(torch.zeros(1))
        # with the normaliser the label also predicts c_k(u), subtracted per
        # program: score_k = sum_j theta_kj(u) q_j(z_k) - c_k(u). It stands in
        # for log Z_k(theta_k(u)), the term a label pays for the size of its
        # coefficients (without it a label predicting a large effect wins for
        # any row pointing its way); the classifier learns its value. Only
        # sum_k c_k(u) enters the logit.
        self.normaliser = normaliser

    @property
    def n_basis(self) -> int:
        return len(self.basis)

    @property
    def n_outputs(self) -> int:
        """Label-network outputs per factor: the coefficients, plus c_k."""
        return len(self.basis) + int(self.normaliser)

    def evidence(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Per-factor evidence [batch, n_factors]; ``lam`` holds theta (and c)."""
        if lam.shape != (*z.shape, self.n_outputs):
            raise ValueError(
                f"lam has shape {tuple(lam.shape)}, expected "
                f"{(*z.shape, self.n_outputs)}"
            )
        e = (lam[..., :self.n_basis] * self.statistics(z)).sum(dim=-1)
        return e - lam[..., self.n_basis] if self.normaliser else e

    def statistics(self, z: torch.Tensor) -> torch.Tensor:
        """[batch, n_factors] -> [batch, n_factors, n_basis]."""
        return torch.stack([BASIS_FUNCTIONS[name](z) for name in self.basis], dim=-1)

    def forward(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Logits for the true/false-label decision.

        Parameters
        ----------
        z
            [batch, n_factors] activities (the group-lasso fit of the row).
        lam
            [batch, n_factors, n_outputs] label-derived coefficients theta (and,
            with the normaliser, c_k last) from
            :class:`~cryptex.models.label_net.FactorizedLabelNet`.

        Returns
        -------
        [batch] logits. Each component contributes a scalar of *evidence*; the
        single real/fake decision is taken on the total. No component votes
        alone, and no pair of components is ever multiplied together.
        """
        return self.evidence(z, lam).sum(dim=1) + self.bias
