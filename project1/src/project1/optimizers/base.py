"""Base class for all optimizers.

Tool used:
    - Copilot autocomplete

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from project1.typing import Array

if TYPE_CHECKING:
    from project1.models import BaseModel


class Optimizer(ABC):
    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma

    @abstractmethod
    def step(self, theta: Array, gradient: Array, t: int) -> Array:
        """Perform one optimization step."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state."""
        raise NotImplementedError

    def optimize(
        self,
        model: BaseModel,
        theta0: Array | None = None,
        max_iter: int = 10000,
        tol: float = 1e-8,
        autodiff: bool = False,
    ) -> OptimizationResult:
        """Optimize a model using this optimizer."""
        if theta0 is None:
            theta = np.zeros(model.X.shape[1])
        else:
            theta = theta0.copy()

        self.reset()

        history = [theta.copy()]
        costs = [model.cost(theta)]

        for t in range(1, max_iter + 1):
            gradient = model.jax_gradient(theta) if autodiff else model.gradient(theta)
            norm = np.linalg.norm(gradient)
            if norm < tol or norm > 1e5:
                break
            theta = self.step(theta, gradient, t)
            c = model.cost(theta)
            if np.isnan(c) or np.isinf(c):
                break
            costs.append(c)
            history.append(theta.copy())

        model.theta = theta  # Update the model's parameters with the optimized values
        return OptimizationResult(history=history, costs=costs, n_iterations=len(history) - 1)


@dataclass
class OptimizationResult:
    history: list[Array]
    costs: list[float]
    n_iterations: int

    @property
    def theta(self) -> Array:
        return self.history[-1]

    @property
    def cost(self) -> float:
        return self.costs[-1]
