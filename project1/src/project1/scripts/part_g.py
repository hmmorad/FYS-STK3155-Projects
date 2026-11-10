"""Script for part g): Lasso regression.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from project1.model_fitter import ModelFitter
from project1.models import Lasso
from project1.optimizers import Adam, GradientDescent, Momentum
from project1.utils import (
    MSE,
    R2,
    fit_polynomial,
    polynomial_matrix,
    predict_polynomial,
    scale_data,
)

if TYPE_CHECKING:
    from typing import Any

    from project1.typing import Array


def gradient_at_zero(x: Array, y: Array, degree: int = 10, lam: float = 0.01) -> dict[str, Array]:
    """Evaluate analytical and JAX Lasso gradients at theta = 0."""
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    model = Lasso(X, y_centered, lam=lam)
    theta_zero = np.zeros(degree)

    return {
        "theta": theta_zero,
        "analytic": model.gradient(theta_zero),
        "jax": np.asarray(model.jax_gradient(theta_zero)),
    }


def compare(
    x: Array,
    y: Array,
    degree: int = 10,
    lam: float = 0.01,
    max_iter: int = 30000,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Compare selected optimizers for Lasso and check against sklearn."""
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    # Let thata:=t, sklearn uses (1 / 2n)||y-Xt||^2 + alpha ||t||_1. Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
    # Our convention is (1 / n)||y-Xt||^2 + lambda ||t||_1.
    sklearn = SklearnLasso(alpha=lam / 2.0, fit_intercept=False, max_iter=max_iter, tol=tol)
    sklearn.fit(X, y_centered)
    sklearn_theta = np.asarray(sklearn.coef_)

    optimizer_specs = {
        r"GD ($\gamma=0.01$)": GradientDescent(gamma=0.01),
        r"Momentum ($\gamma=0.01$)": Momentum(gamma=0.01),
        r"Adam ($\gamma=0.01$)": Adam(gamma=0.01),
    }

    fitter = ModelFitter(model_class=Lasso, model_kwargs={"lam": lam})
    optimize_kwargs_ = {"max_iter": max_iter, "tol": tol}
    results = {}
    for name, optimizer in optimizer_specs.items():
        fitter.set_optimizer(optimizer, optimize_kwargs=optimize_kwargs_)
        model = fitter.fit(X, y_centered)
        if fitter.result is None:
            raise RuntimeError("Optimizer did not return an optimization result.")

        costs = np.asarray(fitter.result.costs)
        results[name] = {
            "theta": model.theta,
            "costs": costs,
            "n_iterations": fitter.result.n_iterations,
            "theta_difference": float(np.linalg.norm(model.theta - sklearn_theta)),
        }

    return {
        "results": results,
        "sklearn_theta": sklearn_theta,
        "gradient_zero": gradient_at_zero(x, y, degree=degree, lam=lam),
    }


def run(
    x: Array,
    y: Array,
    lambdas: Array,
    max_degree: int = 15,
    test_size: float = 0.2,
    seed: int = 2026,
    max_iter: int = 30000,
    tol: float = 1e-6,
) -> dict[str, Array | np.ndarray]:
    """Run Lasso train/test analysis for degree and lambda."""
    fitter = ModelFitter(
        model_class=Lasso,
        optimizer=Adam(gamma=0.1),
        optimize_kwargs={"max_iter": max_iter, "tol": tol},
    )
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
