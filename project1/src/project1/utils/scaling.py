"""
Scaling utilities for matrices.

Tools used:
    - Copilot autocomplete
"""

import numpy as np

from project1.typing import Array


def scale_data(X: Array, y: Array) -> tuple[Array, Array, Array, Array, float]:
    """Scale the features of a matrix and center the target vector.
    Returns:
        X_scaled: The scaled feature matrix.
        mean_X: The mean of each feature in the original matrix.
        std_X: The standard deviation of each feature in the original matrix.
        y_centered: The centered target vector.
        mean_y: The mean of the original target vector.
    """
    X_scaled, mean_X, std_X = scale_features(X)
    y_centered, mean_y = center_y(y)
    return X_scaled, mean_X, std_X, y_centered, mean_y


def scale_features(X: Array) -> tuple[Array, Array, Array]:
    """Scale the features of a matrix to have zero mean and unit variance."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def center_y(y: Array) -> tuple[Array, float]:
    """Center the target vector to have zero mean."""
    mean = np.mean(y)
    y_centered = y - mean
    return y_centered, mean
