"""Loss functions for neural networks.

Tools used:
    - Copilot Autocomplete

"""

from abc import ABC, abstractmethod

import numpy as np

from project2.typing import Array


class Loss(ABC):
    @abstractmethod
    def compute(self, y_pred: Array, y_true: Array) -> float:
        pass


class MSELoss(Loss):  # Half-squared
    def compute(self, y_pred: Array, y_true: Array) -> float:
        n = y_true.shape[0]
        return float(0.5 * np.sum((y_pred - y_true) ** 2) / n)


class BinaryCrossEntropyLoss(Loss):
    def compute(self, y_pred: Array, y_true: Array) -> float:
        eps = 1e-12
        p = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


class CrossEntropyLoss(Loss):
    def compute(self, y_pred: Array, y_true: Array) -> float:
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))
