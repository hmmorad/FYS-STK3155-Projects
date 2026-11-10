"""Script for part b): Ridge regression.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split  # type:ignore

from project1.model_fitter import ModelFitter
from project1.models import Ridge
from project1.utils import MSE, R2, fit_polynomial, predict_polynomial

if TYPE_CHECKING:
    from project1.typing import Array


def run(
    x: Array,
    y: Array,
    lambdas: Array,
    max_degree: int = 15,
    test_size: float = 0.2,
    seed: int = 2026,
) -> dict[str, Array | np.ndarray]:
    """Run Ridge train/test analysis for degree and lambda."""
    fitter = ModelFitter(model_class=Ridge)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    train_mse = np.empty((max_degree, len(lambdas)))
    test_mse = np.empty((max_degree, len(lambdas)))
    train_r2 = np.empty((max_degree, len(lambdas)))
    test_r2 = np.empty((max_degree, len(lambdas)))

    theta = np.full((max_degree, len(lambdas), max_degree), np.nan)

    for j, lam in enumerate(lambdas):
        fitter.set_lambda(float(lam))

        for degree in range(1, max_degree + 1):
            model, x_mean, x_std, y_mean = fit_polynomial(fitter, x_train, y_train, degree)

            y_train_pred = predict_polynomial(fitter, x_train, degree, x_mean, x_std, y_mean)
            y_test_pred = predict_polynomial(fitter, x_test, degree, x_mean, x_std, y_mean)

            i = degree - 1
            train_mse[i, j] = MSE(y_train, y_train_pred)
            test_mse[i, j] = MSE(y_test, y_test_pred)
            train_r2[i, j] = R2(y_train, y_train_pred)
            test_r2[i, j] = R2(y_test, y_test_pred)
            theta[i, j, :degree] = model.theta

    return {
        "degrees": np.arange(1, max_degree + 1),
        "lambdas": np.asarray(lambdas),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "theta": theta,
    }
