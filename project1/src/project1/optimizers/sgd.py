"""Stochastic Gradient Descent (SGD)

Tool used:
    - Copilot autocomplete

"""

import numpy as np

from project1.models.base import BaseModel
from project1.optimizers.base import OptimizationResult, Optimizer


def make_batches(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Create mini-batches from the dataset."""
    idx = rng.permutation(n)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def step_length(t: int, t0: float, t1: float) -> float:
    return t0 / (t + t1)


class SGD:
    def __init__(
        self,
        optimizer: Optimizer | None = None,
        n_epochs: int = 50,
        batch_size: int = 5,
        seed: int = 2026,
        t0: float | None = None,
        t1: float | None = None,
    ):
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.t0 = t0
        self.t1 = t1

    def optimize(self, model: BaseModel, theta0: np.ndarray | None = None) -> OptimizationResult:
        """Optimize a model using Stochastic Gradient Descent (SGD)."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be provided for SGD.")
        self.optimizer.reset()
        if theta0 is None:
            theta = np.zeros(model.X.shape[1])
        else:
            theta = theta0.copy()

        rng = np.random.default_rng(self.seed)
        history = [theta.copy()]
        costs = [model.cost(theta)]

        t = 0
        for _ in range(self.n_epochs):
            batches = make_batches(len(model.y), self.batch_size, rng)
            for batch in batches:
                t += 1
                gradient = model.gradient(theta, batch)

                if self.t0 is not None and self.t1 is not None:
                    self.optimizer.gamma = step_length(t, self.t0, self.t1)

                theta = self.optimizer.step(theta, gradient, t)
                history.append(theta.copy())
                costs.append(model.cost(theta))

        model.theta = theta
        return OptimizationResult(history=history, costs=costs, n_iterations=t)

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def set_t(self, t0: float | None, t1: float | None) -> None:
        self.t0 = t0
        self.t1 = t1

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
