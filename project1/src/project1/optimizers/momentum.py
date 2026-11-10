"""Momentum Optimizer

Tool used:
    - Copilot autocomplete

"""

import numpy as np

from project1.typing import Array

from .base import Optimizer


class Momentum(Optimizer):
    def __init__(self, gamma: float = 0.01, beta: float = 0.9):
        super().__init__(gamma)
        self.beta = beta
        self.v: Array | None = None

    def step(self, theta: Array, gradient: Array, _: int) -> Array:
        if self.v is None:
            self.v = np.zeros_like(theta)

        self.v = self.beta * self.v + self.gamma * gradient
        return theta - self.gamma * self.v

    def reset(self) -> None:
        self.v = None
