"""Script for part f): momentum and adaptive optimizers.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from project1.model_fitter import ModelFitter
from project1.models import OLS
from project1.optimizers import get_optimizer_by_name
from project1.utils.polynomial import polynomial_matrix
from project1.utils.scaling import scale_data

if TYPE_CHECKING:
    from typing import Any

    from project1.models import BaseModel
    from project1.typing import Array


def run_comparison(
    x: Array,
    y: Array,
    degree: int = 10,
    runs: list[tuple[str, float, str]] | None = None,
    max_iter: int = 30000,
    tol: float = 1e-8,
    model_class: type[BaseModel] = OLS,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run selected optimizers and return convergence histories."""
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    reference = model_class(X, y_centered, **(model_kwargs or {}))
    theta_closed = reference.closed_form()
    eigs = reference.hessian_eigs()
    cost_reference = reference.cost(theta_closed)
    gamma_max = 2.0 / eigs.max()
    kappa = eigs.max() / eigs.min()

    if runs is None:
        runs = [
            (
                "GD",
                0.9 * gamma_max,
                r"plain GD, $\gamma=0.9 \gamma_{\max}$,$\gamma_{\max}=$" + f"{gamma_max:.3f}",
            ),
            ("Momentum", 0.9 * gamma_max, r"momentum, $\gamma=0.9 \gamma_{\max}$"),
            ("AdaGrad", 0.1, r"AdaGrad, $\gamma=0.1$"),
            ("RMSProp", 0.1, r"RMSProp, $\gamma=0.1$"),
            ("Adam", 0.1, r"Adam, $\gamma=0.1$"),
        ]

    fitter = ModelFitter(model_class=model_class, model_kwargs=model_kwargs)
    optimize_kwargs_ = {"max_iter": max_iter, "tol": tol}
    results = {}
    for method, gamma, label in runs:
        optimizer_ = get_optimizer_by_name(method, gamma=gamma)
        fitter.set_optimizer(optimizer_, optimize_kwargs=optimize_kwargs_)
        fitter.fit(X, y_centered)
        if fitter.result is None:
            raise RuntimeError("Optimizer did not return an optimization result.")

        history = np.asarray(fitter.result.history)
        costs = np.asarray(fitter.result.costs)
        parameter_error = np.linalg.norm(history - theta_closed, axis=1)
        excess_cost = costs - cost_reference

        results[method] = {
            "label": label,
            "gamma": gamma,
            "history": history,
            "costs": costs,
            "excess_cost": excess_cost,
            "parameter_error": parameter_error,
            "n_iterations": fitter.result.n_iterations,
        }

    return {
        "theta_closed": theta_closed,
        "kappa": float(kappa),
        "results": results,
    }


def learning_rate_sensitivity(
    x: Array,
    y: Array,
    degree: int = 10,
    gammas: Array | None = None,
    target: float = 1e-6,
    max_iter: int = 20000,
    model_class: type[BaseModel] = OLS,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure sensitivity to the initial learning rate."""
    if gammas is None:
        gammas = np.geomspace(1e-3, 1.0, 15)

    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    reference = model_class(X, y_centered, **(model_kwargs or {}))
    theta_closed = reference.closed_form()
    cost_closed = reference.cost(theta_closed)
    eigs = reference.hessian_eigs()
    gamma_max = 2.0 / eigs.max()
    kappa = reference.condition_number()

    methods = ("GD", "Momentum", "AdaGrad", "RMSProp", "Adam")
    iterations = np.full((len(methods), len(gammas)), np.nan)

    fitter = ModelFitter(model_class=model_class, model_kwargs=model_kwargs)
    optimize_kwargs_ = {"max_iter": max_iter}
    for i, method in enumerate(methods):
        for j, gamma in enumerate(gammas):
            fitter.set_optimizer(
                optimizer=get_optimizer_by_name(method, gamma=float(gamma)),
                optimize_kwargs=optimize_kwargs_,
            )
            fitter.fit(X, y_centered)
            if fitter.result is None:
                continue

            excess_cost = np.asarray(fitter.result.costs) - cost_closed
            hit = np.flatnonzero(excess_cost < target)
            if len(hit) > 0:
                iterations[i, j] = hit[0]

    return {
        "gammas": np.asarray(gammas),
        "kappa": float(kappa),
        "methods": methods,
        "iterations": iterations,
        "gamma_max": float(gamma_max),
        "target": target,
    }
