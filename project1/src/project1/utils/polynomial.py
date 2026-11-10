"""Shared helpers for project experiments.

Tools used:
    - Copilot autocomplete
"""

import numpy as np

from project1.model_fitter import ModelFitter
from project1.models import BaseModel
from project1.typing import Array
from project1.utils.scaling import scale_data


def polynomial_matrix(x: Array, degree: int) -> Array:
    """Create the design matrix [x, x^2, ..., x^degree] (without intercept)."""
    return np.column_stack([x**d for d in range(1, degree + 1)])


def fit_polynomial(
    model_fitter: ModelFitter, x: Array, y: Array, degree: int
) -> tuple[BaseModel, Array, Array, float]:
    """Fit a polynomial. The data is scaled before fitting."""
    X = polynomial_matrix(x, degree)
    X, x_mean, x_std, y_centered, y_mean = scale_data(X, y)

    model = model_fitter.fit(X, y_centered)
    return model, x_mean, x_std, y_mean


def predict_polynomial(
    model_fitter: ModelFitter, x: Array, degree: int, x_mean: Array, x_std: Array, y_mean: float
) -> Array:
    """Predict a polynomial.
    Given the training scaling parameters, we scale the input data accordingly.
    """
    X = polynomial_matrix(x, degree)
    X = (X - x_mean) / x_std

    return model_fitter.predict(X) + y_mean
