"""Gradient Descent optimizer implementation for neural networks.

Tools used:
    - Copilot Autocomplete

"""

from project2.network.layer import DenseLayer
from project2.optimizers.base import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, gamma: float = 0.001) -> None:
        super().__init__(gamma)

    def step(self, layers: list[DenseLayer]) -> None:
        for layer in layers:
            if layer.dW is None or layer.db is None:
                raise ValueError("Layer gradients (dW and db) must not be None.")
            layer.W -= self.gamma * layer.dW
            layer.b -= self.gamma * layer.db

    def reset(self) -> None:
        pass
