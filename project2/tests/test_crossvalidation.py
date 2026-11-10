"""Test our cross-validation implementation.

LLM-assisted
------------
Tool: ChatGPT (November 2026)
Role: Wrote the test
Modifications: Small changes to fit our implementation, and fixed typing
Prompt: Gave it the code for our implementation, and cross-validation implementation in Sklearn
in exercisesweek44_solutions.ipynb from https://github.com/EducationalMaterialUiO/MachineLearningUiO/tree/main

"""

from typing import Any

import numpy as np
from project2.network.activations import Identity, ReLU
from project2.optimizers.gradient_descent import GradientDescent
from project2.training.cross_validation import CrossValidation
from project2.training.network_trainer import NetworkTrainer
from project2.typing import Array
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.model_selection import KFold, cross_validate  # type: ignore


def test_cross_validation() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(23, 2))
    y = 2 * X[:, :1] - X[:, 1:] + 0.1 * rng.normal(size=(23, 1))

    settings: dict[str, Any] = dict(
        network_kwargs=dict(n_inputs=2, layers=((3, ReLU), (1, Identity)), seed=42),
        optimizer=GradientDescent,
        optimizer_kwargs=dict(gamma=0.01),
    )
    training = dict(epochs=20, batch_size=None, seed=42)

    # Our cross-validation.
    result = CrossValidation(**settings, n_splits=5, seed=42, scale_features=False).run(
        X, y, trainer_kwargs=training
    )

    # This adapter lets sklearn train our network.
    class SklearnNetwork(RegressorMixin, BaseEstimator):  # type: ignore
        def fit(self, X: Array, y: Array) -> Any:
            self.trainer = NetworkTrainer(**settings, **training)  # type: ignore
            self.trainer.fit(X, y)
            return self

        def predict(self, X: Array) -> Array:
            return self.trainer.predict(X)

    scores = cross_validate(
        SklearnNetwork(),
        X,
        y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        return_train_score=True,
        error_score="raise",
    )

    # Sklearn returns negative MSE; our loss is half the MSE.
    expected = -scores["test_score"] / 2
    expected_train = -scores["train_score"] / 2
    expected_se = np.std(expected, ddof=1) / np.sqrt(5)

    np.testing.assert_allclose(result.fold_scores, expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.mean_score, expected.mean(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.standard_error, expected_se, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.mean_train_score, expected_train.mean(), rtol=1e-7, atol=1e-9)

    print("\nArchitecture: 2 -> 3 (ReLU) -> 1 (Identity)")
    print("5 folds | 20 epochs | GD, learning rate: 0.01")
    print("\nHalf-MSE loss:")
    print(f"{'Fold':<8} {'Our CV':>12} {'Sklearn':>12} {'Abs. diff':>12}")
    for i, (ours, sklearn_score) in enumerate(
        zip(result.fold_scores, expected, strict=True), start=1
    ):
        print(f"{i:<8} {ours:>12.6f} {sklearn_score:>12.6f} {abs(ours - sklearn_score):>12.1e}")

    print(f"\n{'Summary':<20} {'Our CV':>12} {'Sklearn':>12}")
    print(f"{'Mean validation loss':<20} {result.mean_score:>12.6f} {expected.mean():>12.6f}")
    print(f"{'Standard error':<20} {result.standard_error:>12.6f} {expected_se:>12.6f}")
    print(
        f"{'Mean training loss':<20} "
        f"{result.mean_train_score:>12.6f} {expected_train.mean():>12.6f}"
    )
    print("\nAll cross-validation checks passed.")
