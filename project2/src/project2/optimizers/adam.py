"""Adam optimizer implementation for neural networks.

Tools used:
    - Copilot Autocomplete

"""

import numpy as np

from project2.network.layer import DenseLayer
from project2.optimizers.base import Optimizer
from project2.typing import Array


class Adam(Optimizer):
    def __init__(
        self,
        gamma: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(gamma)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: list[tuple[Array, Array]] | None = None
        self.v: list[tuple[Array, Array]] | None = None

    def reset(self) -> None:
        """Reset the optimizer state."""
        self.t = 0
        self.m = None
        self.v = None

    def step(self, layers: list[DenseLayer]) -> None:
        self.t += 1

        if self.m is None or self.v is None:
            self.m = [(np.zeros_like(layer.W), np.zeros_like(layer.b)) for layer in layers]
            self.v = [(np.zeros_like(layer.W), np.zeros_like(layer.b)) for layer in layers]

        for i, layer in enumerate(layers):
            m_W, m_b = self.m[i]
            v_W, v_b = self.v[i]

            if layer.dW is None or layer.db is None:
                raise ValueError("Layer gradients (dW and db) must not be None.")

            # Update moments
            m_W = self.beta1 * m_W + (1 - self.beta1) * layer.dW
            m_b = self.beta1 * m_b + (1 - self.beta1) * layer.db
            v_W = self.beta2 * v_W + (1 - self.beta2) * layer.dW**2
            v_b = self.beta2 * v_b + (1 - self.beta2) * layer.db**2

            self.m[i] = (m_W, m_b)
            self.v[i] = (v_W, v_b)

            # Bias correction
            m_hat_W = m_W / (1 - self.beta1**self.t)
            m_hat_b = m_b / (1 - self.beta1**self.t)
            v_hat_W = v_W / (1 - self.beta2**self.t)
            v_hat_b = v_b / (1 - self.beta2**self.t)

            # Update parameters of the layer
            layer.W -= self.gamma * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.b -= self.gamma * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
