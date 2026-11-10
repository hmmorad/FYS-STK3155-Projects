"""
Base model class

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from project1.typing import Array

if TYPE_CHECKING:
    from typing import Any


class BaseModel:
    def __init__(self, X: Array, y: Array, lam: float = 0.0, penalty: str = ""):
        self.X = X
        self.y = y
        self.lam = lam
        self.penalty = penalty  # "L_1" for Lasso, "" for OLS/Ridge
        self.theta: Array = np.zeros(self.X.shape[1])
        self._jax_gradient = jax.jit(jax.grad(self.cost_jax))

    def closed_form(self) -> Array:
        """Compute the closed-form solution for the model parameters, if available."""
        raise NotImplementedError("Closed-form solution is not available for this model.")

    def predict(self, X_: Array | None = None) -> Array:  # Maybe just change to staticmethod
        """Predict the target values using the model parameters."""
        X_ = self.X if X_ is None else X_
        return X_ @ self.theta

    def cost(self, theta: Array, batch: NDArray[np.int_] | None = None) -> Any:
        """
        Compute the cost function. Can

        OLS: C(theta) = (1/n) ||X theta - y||^2.
        Ridge: C(theta) = (1/n) ||X theta - y||^2 + lam * ||theta||^2.
        Lasso: C(theta) = (1/n) ||X theta - y||^2 + lam * ||theta||_1.
        """
        X = self.X if batch is None else self.X[batch]
        y = self.y if batch is None else self.y[batch]
        n = X.shape[0]

        theta_norm = np.sum(np.abs(theta)) if self.penalty == "L_1" else theta @ theta
        residuals = X @ theta - y
        return np.sum(residuals**2) / n + self.lam * theta_norm

    def cost_jax(self, theta: Array, batch: NDArray[np.int_] | None = None) -> Any:
        """Compute the cost function using JAX."""
        X = jnp.asarray(self.X if batch is None else self.X[batch])
        y = jnp.asarray(self.y if batch is None else self.y[batch])
        n = X.shape[0]

        theta_norm = jnp.sum(jnp.abs(theta)) if self.penalty == "L_1" else jnp.sum(theta**2)
        return jnp.sum((X @ theta - y) ** 2) / n + self.lam * theta_norm

    def jax_gradient(self, theta: Array, batch: NDArray[np.int_] | None = None) -> Any:
        """Compute the gradient of the cost function using JAX."""
        return np.asarray(self._jax_gradient(theta, batch))

    def gradient(self, theta: Array, batch: NDArray[np.int_] | None = None) -> Any:
        """Compute the gradient of the cost function with respect to the model parameters."""
        X = self.X if batch is None else self.X[batch]
        y = self.y if batch is None else self.y[batch]
        n = X.shape[0]

        add_term = self.lam * np.sign(theta) if self.penalty == "L_1" else 2.0 * self.lam * theta
        return (2.0 / n) * X.T @ (X @ theta - y) + add_term

    def hessian_eigs(self) -> Any:
        """Compute the eigenvalues of the Hessian matrix of the cost function for the model."""
        n = self.X.shape[0]
        I = np.eye(self.X.shape[1])  # noqa
        return np.linalg.eigvalsh((2.0 / n) * self.X.T @ self.X + 2.0 * self.lam * I)

    def condition_number(self) -> float:
        """Compute the condition number of the Hessian matrix of the cost function for the model."""
        eigs = self.hessian_eigs()
        return float(eigs.max() / eigs.min())
