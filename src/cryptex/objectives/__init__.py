"""Training objectives: label-swap discrimination and masked reconstruction."""

from .contrastive import (
    DEFAULT_WEIGHTS,
    STRATEGIES,
    StratifiedNegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
)
from .reconstruction import weighted_squared_error

__all__ = [
    "StratifiedNegativeSampler",
    "contrastive_loss",
    "discrimination_accuracy",
    "weighted_squared_error",
    "STRATEGIES",
    "DEFAULT_WEIGHTS",
]
