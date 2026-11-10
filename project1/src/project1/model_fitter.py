"""Combines a model and an optional optimizer to fit the model to data.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from project1.typing import Array

if TYPE_CHECKING:
    from typing import Any

    from project1.models import BaseModel
    from project1.optimizers import OptimizationResult, Optimizer


class ModelFitter:
    def __init__(
        self,
        model_class: type[BaseModel],
        model_kwargs: dict[str, Any] | None = None,
        optimizer: Optimizer | None = None,
        optimize_kwargs: dict[str, Any] | None = None,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.optimizer = optimizer
        self.optimize_kwargs = optimize_kwargs or {}

        self.model: BaseModel | None = None
        self.result: OptimizationResult | None = None

    def fit(self, X: Array, y: Array) -> BaseModel:
        """
        Fit the model to the data using the optimizer if provided.
        Otherwise, use the closed-form solution.
        Returns the fitted model.
        """
        model = self.model_class(X, y, **self.model_kwargs)
        self.result = None
        if self.optimizer is None:
            model.theta = model.closed_form()
        else:
            self.result = self.optimizer.optimize(model, **self.optimize_kwargs)

        self.model = model
        return model

    def predict(self, X: Array) -> Array:
        """Predict using the fitted model."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    def set_lambda(self, lam: float) -> None:
        """Set the regularization parameter for the model."""
        self.model_kwargs["lam"] = lam

    def set_gamma(self, gamma: float) -> None:
        """Set the learning rate for the optimizer."""
        if self.optimizer is not None:
            self.optimizer.gamma = gamma

    def set_optimizer(
        self, optimizer: Optimizer, optimize_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Set the optimizer and optionally its keyword arguments."""
        self.optimizer = optimizer
        self.optimize_kwargs = {} if optimize_kwargs is None else optimize_kwargs
