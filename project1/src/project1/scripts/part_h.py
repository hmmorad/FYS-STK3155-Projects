"""Script for part h): stochastic gradient descent.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from project1.models import OLS, Ridge
from project1.optimizers import SGD, get_optimizer_by_name
from project1.optimizers.sgd import make_batches
from project1.typing import Array
from project1.utils.polynomial import polynomial_matrix
from project1.utils.scaling import scale_data

if TYPE_CHECKING:
    from typing import Any

    from project1.models import BaseModel


def gradient_cloud(
    x: Array,
    y: Array,
    degree: int = 2,
    lam: float = 0.0,
    theta: Array | None = None,
    batch_sizes: tuple[int, ...] = (5, 25),
    n_rounds: int = 40,
    seed: int = 2026,
) -> dict[str, Any]:
    """Generate many minibatch gradients at a fixed parameter value."""
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)

    theta = np.array([0.0, 3.0]) if theta is None else theta

    model = OLS(X, y_centered) if lam == 0.0 else Ridge(X, y_centered, lam=lam)
    full_gradient = model.gradient(theta)
    rng = np.random.default_rng(seed)

    gradients = {}
    for batch_size in batch_sizes:
        batches = []
        for _ in range(n_rounds):
            batches.extend(make_batches(len(y_centered), batch_size, rng))
        gradients[batch_size] = np.array([model.gradient(theta, batch) for batch in batches])

    return {
        "full_gradient": full_gradient,
        "gradients": gradients,
    }


def run_comparison(
    x: Array,
    y: Array,
    degree: int = 10,
    n_epochs: int = 100,
    seed: int = 2026,
    runs: list[tuple[str, dict[str, int], float | None, str]] | None = None,
    model_class: type[BaseModel] = OLS,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare optimizers inside SGD for one minibatch size.
    Args:
    ...
    runs: A list of tuples specifying the optimizer runs.
    schedule: A tuple specifying the learning rate schedule (t0, t1) for SGD.

    Returns:
        A dictionary containing the closed-form solution and the results for each optimizer.
    """
    X = polynomial_matrix(x, degree)
    X, _, _, y_centered, _ = scale_data(X, y)
    model = model_class(X, y_centered, **(model_kwargs or {}))
    theta_closed = model.closed_form()
    cost_closed = model.cost(theta_closed)
    eigs = model.hessian_eigs()
    kappa = model.condition_number()
    gamma_max = 2 / eigs.max()

    if runs is None:
        runs = [
            ("GD", {}, 0.9 * gamma_max, r"full batch, $\gamma=0.9 \gamma_{\max}$"),
            ("GD", {"batch_size": 5}, 0.02, r"SGD, M=5, $\gamma=0.02$"),
            ("GD", {"batch_size": 20}, 0.02, r"SGD, M=20, $\gamma=0.02$"),
            (
                "GD",
                {"batch_size": 5, "t0": 1, "t1": 10},
                None,
                r"SGD, M=5, $\gamma_t = 1/(t+10)$",
            ),
        ]

    results = {}
    sgd = SGD(n_epochs=n_epochs, seed=seed)
    for opt_name, sgd_settings, gamma, label in runs:
        optimizer = get_optimizer_by_name(opt_name, gamma=gamma)
        batch_size_ = sgd_settings.get("batch_size", len(y_centered))

        # Set SGD parameters
        sgd.set_optimizer(optimizer)
        sgd.set_batch_size(batch_size_)
        sgd.set_t(sgd_settings.get("t0"), sgd_settings.get("t1"))

        result = sgd.optimize(model)
        history = np.asarray(result.history)
        costs = np.asarray(result.costs)
        parameter_error = np.linalg.norm(history - theta_closed, axis=1)
        gradient_evaluations = np.arange(len(history)) * batch_size_
        excess_cost = costs - cost_closed

        results[label] = {
            "result": result,
            "parameter_error": parameter_error,
            "gradient_evaluations": gradient_evaluations,
            "excess_cost": excess_cost,
        }

    return {
        "theta_closed": theta_closed,
        "results": results,
        "kappa": kappa,
    }
