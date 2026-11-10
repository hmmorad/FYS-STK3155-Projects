"""AdaGrad Optimizer

Tool used:
    - Copilot autocomplete

"""

import numpy as np

from project1.typing import Array

from .base import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, gamma: float = 0.01, eps: float = 1e-8):
        super().__init__(gamma)
        self.eps = eps
        self.r: Array | None = None

    def step(self, theta: Array, gradient: Array, _: int) -> Array:
        if self.r is None:
            self.r = np.zeros_like(theta)

        self.r += gradient**2
        adjusted_gradient = gradient / (np.sqrt(self.r) + self.eps)
        return theta - self.gamma * adjusted_gradient

    def reset(self) -> None:
        self.r = None
