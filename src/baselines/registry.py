"""Common interface so baselines are swappable in one evaluation harness.

Every baseline exposes the same three things, which is what makes them usable as
baselines rather than one-off notebooks:

    model = get("linear_ica")(n_factors=20).fit(X, perturbations, contexts)
    Z = model.transform(X)      # [n_samples, n_factors]
    B = model.loadings          # [n_factors, n_genes]

``Z`` feeds ``cryptex.interpret.effects.decompose_effects`` and ``B`` feeds
``cryptex.interpret.loadings``, so every method is interpreted and scored
by identical code.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FactorModel(Protocol):
    """Minimal contract for a baseline."""

    def fit(
        self,
        X: np.ndarray,
        perturbations: np.ndarray | None = ...,
        contexts: np.ndarray | None = ...,
    ) -> "FactorModel": ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def loadings(self) -> np.ndarray: ...


_REGISTRY: dict[str, Callable[..., FactorModel]] = {}


def register(name: str) -> Callable[[Callable[..., FactorModel]], Callable[..., FactorModel]]:
    """Class decorator adding a baseline to the registry."""

    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"baseline {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get(name: str) -> Callable[..., FactorModel]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {available()}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
