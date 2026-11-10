import numpy as np

from project1.typing import Array


def MSE(y_data: Array, y_model: Array) -> float:
    """Calculates the MSE"""
    return np.mean((y_data - y_model) ** 2)


def R2(y_data: Array, y_model: Array) -> float:
    """Calculates the R^2."""
    ss_res = np.sum((y_data - y_model) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1 - ss_res / ss_tot
