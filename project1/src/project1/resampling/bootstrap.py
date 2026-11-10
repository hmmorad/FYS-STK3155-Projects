"""Bootstrap resampling for model evaluation.

Tools used:
    - Copilot autocomplete
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils import resample  # type: ignore

from project1.typing import Array
from project1.utils import scale_data
from project1.utils.metrics import MSE
from project1.utils.polynomial import polynomial_matrix

if TYPE_CHECKING:
    from project1.model_fitter import ModelFitter


@dataclass
class BootstrapResult:
    predictions: Array
    error: float
    bias2: float
    variance: float


class Bootstrap:
    def __init__(
        self,
        x: Array,
        y: Array,
        model_fitter: ModelFitter,
        n_bootstraps: int = 100,
        test_size: float = 0.2,
        seed: int = 2026,
    ):
        self.x = x
        self.y = y
        self.model_fitter = model_fitter
        self.n_bootstraps = n_bootstraps
        self.test_size = test_size
        self.seed = seed

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed
        )

    def run(self, degree: int) -> BootstrapResult:
        """Run bootstrap for one polynomial degree."""
        X_train = polynomial_matrix(self.x_train, degree)
        X_test = polynomial_matrix(self.x_test, degree)
        predictions = np.empty((self.n_bootstraps, len(X_test)))

        # Scale using training data only.
        X_train, x_mean, x_std, y_train_centered, y_mean = scale_data(X_train, self.y_train)
        X_test = (X_test - x_mean) / x_std

        rnd_state = np.random.RandomState(self.seed)
        for b in range(self.n_bootstraps):
            X_boot, y_boot = resample(X_train, y_train_centered, random_state=rnd_state)  # pyright: ignore[reportGeneralTypeIssues]

            _ = self.model_fitter.fit(X_boot, y_boot)
            predictions[b] = self.model_fitter.predict(X_test) + y_mean  # add intercept

        # Treating as if f(x) is unknown
        mean_prediction = np.mean(predictions, axis=0)
        error = MSE(self.y_test, predictions)
        bias2 = np.mean((self.y_test - mean_prediction) ** 2)
        variance = np.mean(np.var(predictions, axis=0))
        return BootstrapResult(predictions=predictions, error=error, bias2=bias2, variance=variance)

    def bias_variance_sweep(self, max_degree: int) -> tuple[Array, Array, Array]:
        """Run the bias-variance sweep for polynomial degrees."""
        error = np.empty(max_degree)
        bias2 = np.empty(max_degree)
        variance = np.empty(max_degree)

        for degree in range(1, max_degree + 1):
            result = self.run(degree)
            i = degree - 1
            error[i] = result.error
            bias2[i] = result.bias2
            variance[i] = result.variance

        return error, bias2, variance
