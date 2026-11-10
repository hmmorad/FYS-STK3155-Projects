"""Script for part a): OLS regression.

Tools used:
    - Github copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split  # type:ignore

from project1.model_fitter import ModelFitter
from project1.models import OLS
from project1.utils import MSE, R2, fit_polynomial, predict_polynomial

if TYPE_CHECKING:
    from project1.typing import Array


def run(
    x: Array, y: Array, max_degree: int = 15, test_size: float = 0.2, seed: int = 2026
) -> dict[str, Array | np.ndarray]:
    """Run OLS train/test analysis for polynomial degrees."""
    fitter = ModelFitter(model_class=OLS)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    train_mse = np.empty(max_degree)
    test_mse = np.empty(max_degree)
    train_r2 = np.empty(max_degree)
    test_r2 = np.empty(max_degree)

    # Row = degree, column = fitted coefficient theta_j.
    theta = np.full((max_degree, max_degree), np.nan)

    for degree in range(1, max_degree + 1):
        i = degree - 1
        model, x_mean, x_std, y_mean = fit_polynomial(fitter, x_train, y_train, degree)

        y_train_pred = predict_polynomial(fitter, x_train, degree, x_mean, x_std, y_mean)
        train_mse[i] = MSE(y_train, y_train_pred)
        train_r2[i] = R2(y_train, y_train_pred)

        y_test_pred = predict_polynomial(fitter, x_test, degree, x_mean, x_std, y_mean)
        test_mse[i] = MSE(y_test, y_test_pred)
        test_r2[i] = R2(y_test, y_test_pred)

        theta[i, :degree] = model.theta
    return {
        "degrees": np.arange(1, max_degree + 1),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "theta": theta,
    }
