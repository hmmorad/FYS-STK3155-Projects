"""Gradient Descent

Tool used:
    - Copilot autocomplete

"""

from project1.typing import Array

from .base import Optimizer


class GradientDescent(Optimizer):
    def step(self, theta: Array, gradient: Array, _: int) -> Array:
        return theta - self.gamma * gradient

    def reset(self) -> None:
        pass
