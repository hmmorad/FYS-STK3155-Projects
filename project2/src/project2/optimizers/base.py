"""Base class for all optimizers.

Tool used:
    - Copilot autocomplete

"""

from abc import ABC, abstractmethod

from project2.network.layer import DenseLayer


class Optimizer(ABC):
    def __init__(self, gamma: float = 0.01) -> None:
        self.gamma = gamma  # Learning rate for the optimizer

    @abstractmethod
    def step(self, layers: list[DenseLayer]) -> None:
        """Update the parameters of all layers."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state."""
        pass

    def set_learning_rate(self, gamma: float) -> None:
        """Set the learning rate for the optimizer."""
        self.gamma = gamma
