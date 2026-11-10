"""Script for part d): cross-validation.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from project1.model_fitter import ModelFitter
from project1.models import OLS, Ridge
from project1.resampling import CrossValidation

if TYPE_CHECKING:
    from project1.typing import Array


def run_ols_cv(x: Array, y: Array, max_degree: int, n_splits: int, seed: int) -> Array:
    """Run OLS CV over polynomial degree."""
    cv = CrossValidation(
        x, y, model_fitter=ModelFitter(model_class=OLS), n_splits=n_splits, seed=seed
    )
    return cv.degree_sweep(max_degree)


def run_ridge_cv(
    x: Array, y: Array, lambdas: Array, max_degree: int, n_splits: int, seed: int
) -> Array:
    """Run Ridge CV over degree and lambda."""
    cv = CrossValidation(
        x, y, model_fitter=ModelFitter(model_class=Ridge), n_splits=n_splits, seed=seed
    )
    return cv.grid_sweep(lambdas=lambdas, max_degree=max_degree)


def run(
    x: Array,
    y: Array,
    lambdas: Array,
    max_degree: int = 15,
    seed: int = 2026,
    test_size: float = 0.2,
) -> dict[str, Array | np.ndarray]:
    """Run 5- and 10-fold CV for OLS and Ridge."""
    # We only use the training set for cross-validation.
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, random_state=seed)

    ols_mse_5 = run_ols_cv(x_train, y_train, max_degree, 5, seed)
    ols_mse_10 = run_ols_cv(x_train, y_train, max_degree, 10, seed)

    ridge_mse_5 = run_ridge_cv(x_train, y_train, lambdas, max_degree, 5, seed)
    ridge_mse_10 = run_ridge_cv(x_train, y_train, lambdas, max_degree, 10, seed)

    return {
        "degrees": np.arange(1, max_degree + 1),
        "lambdas": np.asarray(lambdas),
        "ols_mse_5": ols_mse_5,
        "ols_mse_10": ols_mse_10,
        "ridge_mse_5": ridge_mse_5,
        "ridge_mse_10": ridge_mse_10,
    }
