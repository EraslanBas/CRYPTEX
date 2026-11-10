"""Decompose factor activations into perturbation, context and interaction.

    z_{p,c}  =  grand_mean  +  mu_p  +  gamma_c  +  delta_pc

* ``mu_p``     perturbation main effect -- which factors it moves, context-free.
* ``gamma_c``  context main effect -- what the drug does on its own.
* ``delta_pc`` the interaction: where a perturbation's effect on a factor
               *depends* on the drug. This is the scientific payload of a
               chemogenetic screen.

All three live in the same factor space and share the same global loadings, so
each is interpretable through the same ``B``.

Caveat with one observation per pair: a fully flexible interaction has as many
free cells as there are observations, so ``delta_pc`` is only estimable if it is
constrained (low-rank or sparse) or if there are replicates per pair. With
pseudo-replicates (``data.pseudoreplicates``) it is identified directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EffectDecomposition:
    grand_mean: np.ndarray  # [n_factors]
    mu: pd.DataFrame  # [n_perturbations, n_factors]
    gamma: pd.DataFrame  # [n_contexts, n_factors]
    delta: pd.DataFrame  # [n_pairs, n_factors], MultiIndex (perturbation, context)

    def variance_shares(self) -> pd.Series:
        """Fraction of total factor-activation variance in each term.

        A useful first read on the screen: if ``delta`` carries almost nothing,
        perturbation effects are context-independent and the chemogenetic design
        is not buying anything beyond the main effects.
        """
        parts = {
            "perturbation": np.var(self.mu.to_numpy()),
            "context": np.var(self.gamma.to_numpy()),
            "interaction": np.var(self.delta.to_numpy()),
        }
        total = sum(parts.values())
        return pd.Series({k: v / total for k, v in parts.items()})


def decompose_effects(
    Z: np.ndarray,
    perturbations: np.ndarray,
    contexts: np.ndarray,
) -> EffectDecomposition:
    """Two-way decomposition of factor activations.

    Uses unweighted cell means, so it is exact for a balanced design and an
    approximation otherwise. Replicates of the same ``(p, c)`` pair are averaged
    first.
    """
    Z = np.asarray(Z, dtype=np.float64)
    frame = pd.DataFrame(Z, columns=[f"factor_{k}" for k in range(Z.shape[1])])
    frame["perturbation"] = np.asarray(perturbations)
    frame["context"] = np.asarray(contexts)
    factor_cols = [c for c in frame.columns if c.startswith("factor_")]

    cell = frame.groupby(["perturbation", "context"], observed=True)[factor_cols].mean()
    grand = cell.to_numpy().mean(axis=0)

    mu = cell.groupby("perturbation", observed=True).mean() - grand
    gamma = cell.groupby("context", observed=True).mean() - grand

    idx = cell.index
    delta = (
        cell
        - grand
        - mu.reindex(idx.get_level_values("perturbation")).to_numpy()
        - gamma.reindex(idx.get_level_values("context")).to_numpy()
    )
    return EffectDecomposition(grand_mean=grand, mu=mu, gamma=gamma, delta=delta)
