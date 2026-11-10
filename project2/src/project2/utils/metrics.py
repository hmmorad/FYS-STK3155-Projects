"""Metrics for evaluating model performance.

Tools used:
    - Copilot Autocomplete

"""

import numpy as np

from project2.network.activations import Activation, Softmax
from project2.typing import Array


def accuracy(y_pred: Array, y_val: Array, last_activation: Activation) -> float:
    """Accuracy for multiclass softmax or binary sigmoid outputs."""
    if isinstance(last_activation, Softmax):
        predicted = np.argmax(y_pred, axis=1)
        actual = np.asarray(y_val)

        if actual.ndim == 2 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)  # type: ignore
        else:
            actual = actual.ravel()
    else:
        predicted = np.asarray(y_pred).ravel() >= 0.5  # type: ignore
        actual = np.asarray(y_val).ravel() >= 0.5  # type: ignore

    return float(np.mean(predicted == actual))


def sd_error_binomial(p: float, n: int) -> float:
    """Approximate standard error of a binomial proportion."""
    return float(np.sqrt(p * (1 - p) / n))
