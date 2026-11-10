"""RMSProp Optimizer

Tool used:
    - Copilot autocomplete
"""

import numpy as np

from project2.network.layer import DenseLayer
from project2.optimizers.base import Optimizer
from project2.typing import Array


class RMSProp(Optimizer):
    def __init__(
        self,
        gamma: float = 0.001,
        rho: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(gamma)
        self.rho = rho
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        """Reset the optimizer state."""
        self.r: list[tuple[Array, Array]] | None = None

    def step(self, layers: list[DenseLayer]) -> None:
        if self.r is None:
            self.r = [(np.zeros_like(layer.W), np.zeros_like(layer.b)) for layer in layers]

        for i, layer in enumerate(layers):
            r_W, r_b = self.r[i]
            if layer.dW is None or layer.db is None:
                raise ValueError("Layer gradients (dW and db) must not be None.")

            # Update moving average
            r_W = self.rho * r_W + (1 - self.rho) * layer.dW**2
            r_b = self.rho * r_b + (1 - self.rho) * layer.db**2

            self.r[i] = (r_W, r_b)

            # Update parameters of the layer
            layer.W -= self.gamma * layer.dW / (np.sqrt(r_W) + self.epsilon)
            layer.b -= self.gamma * layer.db / (np.sqrt(r_b) + self.epsilon)
