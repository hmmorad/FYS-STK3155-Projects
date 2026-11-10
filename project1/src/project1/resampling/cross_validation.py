"""K-fold cross-validation for model evaluation.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import KFold  # type: ignore

from project1.typing import Array
from project1.utils import scale_data
from project1.utils.metrics import MSE
from project1.utils.polynomial import polynomial_matrix

if TYPE_CHECKING:
    from project1.model_fitter import ModelFitter


@dataclass
class CrossValidationResult:
    fold_mse: Array
    mean_mse: float

    @property
    def standard_error(self) -> float:
        """Standard error of the mean fold MSE."""
        return float(np.std(self.fold_mse, ddof=1) / np.sqrt(len(self.fold_mse)))


class CrossValidation:
    def __init__(
        self, x: Array, y: Array, model_fitter: ModelFitter, n_splits: int = 5, seed: int = 2026
    ):
        self.x = x
        self.y = y
        self.model_fitter = model_fitter
        self.n_splits = n_splits
        self.seed = seed

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.folds = list(kfold.split(self.x))

    def run(self, degree: int) -> CrossValidationResult:
        """Run cross-validation for one polynomial degree."""
        fold_mse = np.empty(self.n_splits)

        for fold, (train_idx, test_idx) in enumerate(self.folds):
            x_train, y_train = self.x[train_idx], self.y[train_idx]
            x_test, y_test = self.x[test_idx], self.y[test_idx]

            X_train = polynomial_matrix(x_train, degree)
            X_test = polynomial_matrix(x_test, degree)

            # Scale only the training fold and apply train scale on the test fold
            X_train, x_mean, x_std, y_train_centered, y_mean = scale_data(X_train, y_train)
            X_test = (X_test - x_mean) / x_std

            self.model_fitter.fit(X_train, y_train_centered)
            predictions = self.model_fitter.predict(X_test) + y_mean
            fold_mse[fold] = MSE(y_test, predictions)

        return CrossValidationResult(fold_mse=fold_mse, mean_mse=np.mean(fold_mse))

    def degree_sweep(self, max_degree: int) -> Array:
        """Evaluate CV MSE for polynomial degrees 1 through max_degree."""
        mse = np.empty(max_degree)
        for degree in range(1, max_degree + 1):
            result = self.run(degree)
            mse[degree - 1] = result.mean_mse
        return mse

    def lambda_sweep(self, lambdas: Array, degree: int) -> Array:
        """Evaluate CV MSE for a range of lambda values."""
        mse = np.empty(len(lambdas))
        for i, lam in enumerate(lambdas):
            self.model_fitter.set_lambda(lam)
            mse[i] = self.run(degree).mean_mse
        return mse

    def grid_sweep(self, lambdas: Array, max_degree: int) -> Array:
        """Evaluate CV MSE. For each polynomial degree, evaluate over all lambda."""
        mse = np.empty((max_degree, len(lambdas)))
        for degree in range(1, max_degree + 1):
            degree_mse = self.lambda_sweep(lambdas, degree)
            mse[degree - 1] = degree_mse
        return mse
