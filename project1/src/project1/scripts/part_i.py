"""Script for part i): final OLS, Ridge and Lasso CV comparison.

Tools used:
    - Copilot autocomplete
    - ChatGPT (October 2026)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from project1.model_fitter import ModelFitter
from project1.models import OLS, Lasso, Ridge
from project1.optimizers import Adam
from project1.resampling import CrossValidation
from project1.typing import Array
from project1.utils import MSE, R2, polynomial_matrix, scale_data

if TYPE_CHECKING:
    from typing import Any


def _fit_and_test(
    model_fitter: ModelFitter,
    degree: int,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
) -> tuple[float, float, Array]:
    """Fit on all training data and evaluate on the test data."""
    X_train, X_test = polynomial_matrix(x_train, degree), polynomial_matrix(x_test, degree)

    X_train, x_mean, x_std, y_train_centered, y_mean = scale_data(X_train, y_train)
    X_test = (X_test - x_mean) / x_std

    model = model_fitter.fit(X_train, y_train_centered)
    y_test_pred = model_fitter.predict(X_test) + y_mean
    return MSE(y_test, y_test_pred), R2(y_test, y_test_pred), model.theta.copy()


def _select_one_se_1d(mse: Array, se: Array) -> tuple[int, int]:
    """Select the simplest degree within one SE of the minimum.
    (OLS)

    LLM-assisted
    ------------
    Tool: ChatGPT (October 2026)
    Role: Assist in selecting the simplest degree within one standard error of the minimum MSE.
    Modifications: None

    """
    min_index = int(np.argmin(mse))
    threshold = mse[min_index] + se[min_index]
    candidates = np.flatnonzero(mse <= threshold)
    # Smaller degree = simpler model.
    selected_index = int(candidates[0])
    return min_index, selected_index


def _select_one_se_2d(
    mse: Array, se: Array, lambdas: Array
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select the simplest (degree, lambda) pair within one SE of the minimum.

    Simplicity is defined as:
    1. lower polynomial degree;
    2. for the same degree, larger lambda.
    (Ridge, Lasso)

    LLM-assisted
    ------------
    Tool: ChatGPT (October 2026)
    Role: Assist in selecting the simplest degree
    Modifications: None

    """
    min_index = np.unravel_index(np.argmin(mse), mse.shape)
    threshold = mse[min_index] + se[min_index]

    candidates = np.argwhere(mse <= threshold)

    selected_index = min(
        candidates,
        key=lambda index: (
            index[0],  # lower degree
            -lambdas[index[1]],  # larger lambda
        ),
    )

    return (
        (int(min_index[0]), int(min_index[1])),
        (int(selected_index[0]), int(selected_index[1])),
    )


def run(
    x: Array,
    y: Array,
    lambdas: Array,
    max_degree: int = 15,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 2026,
) -> dict[str, Any]:
    """Select models using CV on training data and evaluate on an untouched test set.

    LLM-assisted
    ------------
    Tool: ChatGPT (October 2026)
    Role: Assist in selecting the simplest degree within one standard error of the minimum MSE.
    Modifications: None

    """
    degrees = np.arange(1, max_degree + 1)

    # The test set is never used during CV.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    #### OLS
    ols_fitter = ModelFitter(model_class=OLS)
    ols_cv = CrossValidation(
        x_train, y_train, model_fitter=ols_fitter, n_splits=n_splits, seed=seed
    )

    ols_mse = np.empty(max_degree)
    ols_se = np.empty(max_degree)

    for degree in degrees:
        result = ols_cv.run(int(degree))
        i = degree - 1
        ols_mse[i] = result.mean_mse
        ols_se[i] = result.standard_error

    ols_min_index, ols_index = _select_one_se_1d(ols_mse, ols_se)
    ols_degree = int(degrees[ols_index])

    ols_selected_cv = ols_cv.run(ols_degree)

    ols_test_mse, ols_test_r2, ols_theta = _fit_and_test(
        ols_fitter, ols_degree, x_train, y_train, x_test, y_test
    )

    #### Ridge
    ridge_fitter = ModelFitter(model_class=Ridge)

    ridge_cv = CrossValidation(
        x_train, y_train, model_fitter=ridge_fitter, n_splits=n_splits, seed=seed
    )

    ridge_mse = np.empty((max_degree, len(lambdas)))
    ridge_se = np.empty((max_degree, len(lambdas)))

    for degree in degrees:
        for j, lam in enumerate(lambdas):
            ridge_fitter.set_lambda(float(lam))
            result = ridge_cv.run(int(degree))

            i = degree - 1
            ridge_mse[i, j] = result.mean_mse
            ridge_se[i, j] = result.standard_error

    ridge_min_index, ridge_index = _select_one_se_2d(ridge_mse, ridge_se, lambdas)

    ridge_degree = int(degrees[ridge_index[0]])
    ridge_lambda = float(lambdas[ridge_index[1]])

    ridge_fitter.set_lambda(ridge_lambda)
    ridge_selected_cv = ridge_cv.run(ridge_degree)

    ridge_test_mse, ridge_test_r2, ridge_theta = _fit_and_test(
        ridge_fitter, ridge_degree, x_train, y_train, x_test, y_test
    )

    #### Lasso
    lasso_fitter = ModelFitter(
        model_class=Lasso,
        optimizer=Adam(),
        optimize_kwargs={"max_iter": 10000, "tol": 1e-8, "autodiff": True},
    )

    lasso_cv = CrossValidation(
        x_train, y_train, model_fitter=lasso_fitter, n_splits=n_splits, seed=seed
    )

    lasso_mse = np.empty((max_degree, len(lambdas)))
    lasso_se = np.empty((max_degree, len(lambdas)))

    for degree in degrees:
        for j, lam in enumerate(lambdas):
            lasso_fitter.set_lambda(float(lam))
            result = lasso_cv.run(int(degree))

            i = degree - 1
            lasso_mse[i, j] = result.mean_mse
            lasso_se[i, j] = result.standard_error

    lasso_min_index, lasso_index = _select_one_se_2d(lasso_mse, lasso_se, lambdas)

    lasso_degree = int(degrees[lasso_index[0]])
    lasso_lambda = float(lambdas[lasso_index[1]])

    lasso_fitter.set_lambda(lasso_lambda)
    lasso_selected_cv = lasso_cv.run(lasso_degree)

    lasso_test_mse, lasso_test_r2, lasso_theta = _fit_and_test(
        lasso_fitter, lasso_degree, x_train, y_train, x_test, y_test
    )

    return {
        "degrees": degrees,
        "lambdas": np.asarray(lambdas),
        "ols_mse": ols_mse,
        "ols_se": ols_se,
        "ridge_mse": ridge_mse,
        "ridge_se": ridge_se,
        "lasso_mse": lasso_mse,
        "lasso_se": lasso_se,
        "best_models": {
            "OLS": {
                "degree": ols_degree,
                "lambda": np.nan,
                "cv_mse": float(ols_selected_cv.mean_mse),
                "standard_error": float(ols_selected_cv.standard_error),
                "test_mse": ols_test_mse,
                "test_r2": ols_test_r2,
                "theta": ols_theta,
                "minimum_degree": int(degrees[ols_min_index]),
            },
            "Ridge": {
                "degree": ridge_degree,
                "lambda": ridge_lambda,
                "cv_mse": float(ridge_selected_cv.mean_mse),
                "standard_error": float(ridge_selected_cv.standard_error),
                "test_mse": ridge_test_mse,
                "test_r2": ridge_test_r2,
                "theta": ridge_theta,
                "minimum_degree": int(degrees[ridge_min_index[0]]),
                "minimum_lambda": float(lambdas[ridge_min_index[1]]),
            },
            "Lasso": {
                "degree": lasso_degree,
                "lambda": lasso_lambda,
                "cv_mse": float(lasso_selected_cv.mean_mse),
                "standard_error": float(lasso_selected_cv.standard_error),
                "test_mse": lasso_test_mse,
                "test_r2": lasso_test_r2,
                "theta": lasso_theta,
                "minimum_degree": int(degrees[lasso_min_index[0]]),
                "minimum_lambda": float(lambdas[lasso_min_index[1]]),
            },
        },
    }
