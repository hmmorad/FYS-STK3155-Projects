"""Script for part e): gradient descent and automatic differentiation.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from project1.model_fitter import ModelFitter
from project1.models import OLS, Ridge
from project1.optimizers import GradientDescent
from project1.utils.polynomial import polynomial_matrix
from project1.utils.scaling import scale_data

if TYPE_CHECKING:
    from typing import Any

    from project1.models import BaseModel
    from project1.typing import Array


def gradient_check(
    x: Array, y: Array, degree: int = 10, lam: float = 0.01, seed: int = 2026
) -> dict[str, float]:
    """Compare analytical and JAX gradients for OLS and Ridge."""
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    rng = np.random.default_rng(seed)
    theta = rng.normal(size=degree)

    ols = OLS(X, y_centered)
    ridge = Ridge(X, y_centered, lam=lam)

    return {
        "ols_max_abs_error": float(
            np.max(np.abs(ols.gradient(theta) - np.asarray(ols.jax_gradient(theta))))
        ),
        "ridge_max_abs_error": float(
            np.max(np.abs(ridge.gradient(theta) - np.asarray(ridge.jax_gradient(theta))))
        ),
    }


def run_gd(
    X: Array,
    y: Array,
    max_iter: int,
    tol: float = 1e-8,
    autodiff: bool = False,
    model_class: type[BaseModel] = OLS,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run gradient descent and compare with the closed-form solution."""
    reference_model = model_class(X, y, **(model_kwargs or {}))
    theta_closed = reference_model.closed_form()
    cost_closed = reference_model.cost(theta_closed)
    eigs = reference_model.hessian_eigs()
    gamma_max = 2.0 / eigs.max()
    gamma_best = 2.0 / (eigs.max() + eigs.min())

    fitter = ModelFitter(
        model_class=model_class,
        model_kwargs=model_kwargs,
        optimizer=GradientDescent(),
        optimize_kwargs={"max_iter": max_iter, "tol": tol, "autodiff": autodiff},
    )
    output: dict[str, Any] = {}

    for gamma, gamma_label in [
        (0.01, r"\gamma=0.01"),
        (0.99 * gamma_max, r"\gamma=0.99 \gamma_{\max}. \gamma_{\max}=" + f"{gamma_max:.3f}"),
        (gamma_best, r"\gamma=\gamma^*"),
        (1.02 * gamma_max, r"\gamma=1.02 \gamma_{\max} (\text{diverges})"),
    ]:
        fitter.set_gamma(gamma)
        fitter.fit(X, y)

        if fitter.result is None:
            raise RuntimeError("Gradient descent did not return a result.")

        history = np.asarray(fitter.result.history)
        parameter_error = np.linalg.norm(history - theta_closed, axis=1)
        output[gamma_label] = {
            "history": history,
            "costs": np.asarray(fitter.result.costs),
            "excess_cost": np.asarray(fitter.result.costs) - cost_closed,
            "parameter_error": parameter_error,
            "n_iterations": fitter.result.n_iterations,
            "parameter_error_final": float(parameter_error[-1]),
            "gamma": gamma,
        }

    return output


def run(
    x: Array,
    y: Array,
    degree: int = 10,
    lam: float = 0.01,
    max_iter: int = 30000,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Run gradient descent for OLS and Ridge models. Analytic and autodiff gradients.
    Returns a dict on the form:
    {
        "OLS": {
            "analytic": {...},
            "autodiff": {...}
        },
    }

    """
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    models = {"OLS": (OLS, {}), "Ridge": (Ridge, {"lam": lam})}
    output: dict[str, Any] = {}

    for name, (model_class, model_kwargs) in models.items():
        kappa = model_class(X, y_centered, **(model_kwargs or {})).condition_number()
        output[name] = {
            "kappa": kappa,
            "analytic": run_gd(X, y_centered, max_iter, tol, False, model_class, model_kwargs),
            "autodiff": run_gd(X, y_centered, max_iter, tol, True, model_class, model_kwargs),
        }

    return output
