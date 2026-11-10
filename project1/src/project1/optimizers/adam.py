"""Adam Optimizer

Tool used:
    - Copilot autocomplete

"""

import numpy as np

from project1.optimizers.base import Optimizer
from project1.typing import Array


class Adam(Optimizer):
    def __init__(
        self, gamma: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ):
        super().__init__(gamma)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Array | None = None
        self.r: Array | None = None

    def step(self, theta: Array, gradient: Array, t: int) -> Array:
        if self.m is None:
            self.m = np.zeros_like(theta)
        if self.r is None:
            self.r = np.zeros_like(theta)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.r = self.beta2 * self.r + (1 - self.beta2) * gradient**2

        m_hat = self.m / (1 - self.beta1**t)
        v_hat = self.r / (1 - self.beta2**t)

        adjusted_gradient = m_hat / (np.sqrt(v_hat) + self.eps)
        return theta - self.gamma * adjusted_gradient

    def reset(self) -> None:
        self.m = None
        self.r = None
