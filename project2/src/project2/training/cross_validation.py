from dataclasses import dataclass
from typing import Any

import numpy as np
from project1.utils import scale_data  # From project1
from project2.optimizers.base import Optimizer
from project2.training.network_trainer import NetworkTrainer
from project2.typing import Array
from sklearn.model_selection import KFold  # type: ignore


@dataclass
class CrossValidationResult:
    fold_scores: list[float]
    mean_score: float
    standard_error: float
    mean_train_score: float = np.nan


class CrossValidation:
    def __init__(
        self,
        network_kwargs: dict[str, Any],
        optimizer: type[Optimizer],
        optimizer_kwargs: dict[str, Any] | None = None,
        n_splits: int = 5,
        seed: int = 2026,
        scale_features: bool = False,
    ) -> None:
        self.network_kwargs = network_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.n_splits = n_splits
        self.seed = seed
        self.scale_features = scale_features

    def run(
        self, X: Array, y: Array, trainer_kwargs: dict[str, Any] | None = None
    ) -> CrossValidationResult:
        """Run cross-validation on the given data and return the results."""
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        fold_scores = []
        train_scores = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            y_mean = 0.0
            # Fit scaling on training data only
            if self.scale_features:
                # Scale only the training fold and apply train scale on the test fold
                X_train, x_mean, x_std, y_train, y_mean = scale_data(X_train, y_train)
                X_val = (X_val - x_mean) / x_std

            # Create a fresh trainer for this fold
            trainer = NetworkTrainer(
                network_kwargs=self.network_kwargs,
                optimizer=self.optimizer,
                optimizer_kwargs=self.optimizer_kwargs,
                **(trainer_kwargs or {}),
            )

            # Train and evaluate
            trainer.fit(X_train, y_train)
            train_prediction = trainer.predict(X_train)
            train_score = trainer.network.loss.compute(train_prediction, y_train)
            train_scores.append(float(train_score))

            y_pred = trainer.predict(X_val) + y_mean  # Add back intercept
            score = trainer.network.loss.compute(y_pred, y_val)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores, keepdims=False))
        standard_error = float(np.std(fold_scores, ddof=1) / np.sqrt(self.n_splits))

        return CrossValidationResult(
            fold_scores=fold_scores,
            mean_score=mean_score,
            standard_error=standard_error,
            mean_train_score=float(np.mean(train_scores)),
        )

    def lambda_gamma_grid(
        self,
        X: Array,
        y: Array,
        gamma_values: list[float],
        lambda_values: list[float],
        reg_type: str = "L1",
    ) -> dict[tuple[float, float], CrossValidationResult]:
        """Perform a grid search over lambda and gamma values."""
        if reg_type not in ["L1", "L2"]:
            raise ValueError("reg_type must be 'L1' or 'L2'")
        reg = "lmbd1" if reg_type == "L1" else "lmbd2"

        results = {}
        for lmbd in lambda_values:
            for gamma in gamma_values:
                self.network_kwargs[reg] = lmbd
                self.optimizer_kwargs["gamma"] = gamma
                results[(lmbd, gamma)] = self.run(X, y)
        return results
