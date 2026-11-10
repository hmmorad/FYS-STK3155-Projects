from __future__ import annotations

from typing import TYPE_CHECKING

from .adagrad import AdaGrad
from .adam import Adam
from .base import OptimizationResult, Optimizer
from .gradient_descent import GradientDescent
from .momentum import Momentum
from .rmsprop import RMSProp
from .sgd import SGD

if TYPE_CHECKING:
    from typing import Any


def get_optimizer_by_name(method: str, **kwargs: Any) -> Optimizer:
    """Construct an optimizer by name."""
    optimizers: dict[str, type[Optimizer]] = {
        "GD": GradientDescent,
        "Momentum": Momentum,
        "AdaGrad": AdaGrad,
        "RMSProp": RMSProp,
        "Adam": Adam,
    }
    if method not in optimizers:
        raise ValueError(f"Unknown optimizer: {method}")
    return optimizers[method](**kwargs)


__all__ = [
    "OptimizationResult",
    "Optimizer",
    "GradientDescent",
    "Momentum",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "SGD",
    "get_optimizer_by_name",
]
