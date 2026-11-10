"""Part b: Runge regression with neural networks and gradient optimizers.

Tools used:
    - Copilot autocomplete

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from project1.model_fitter import ModelFitter
from project1.models.ols import OLS
from project1.utils import scale_data
from project1.utils.metrics import MSE, R2
from project1.utils.polynomial import fit_polynomial, predict_polynomial
from sklearn.neural_network import MLPRegressor  # type: ignore

from project2.network.activations import Identity, Sigmoid
from project2.optimizers.adam import Adam
from project2.optimizers.gradient_descent import GradientDescent
from project2.optimizers.rmsprop import RMSProp
from project2.training.network_trainer import NetworkTrainer

if TYPE_CHECKING:
    from typing import Any

    from project2.typing import Array

OPTIMIZERS: dict[str, type] = {"GD": GradientDescent, "RMSProp": RMSProp, "Adam": Adam}


def fit_comparison(
    x_train: Array,
    y_train: Array,
    x_grid: Array,
    hidden_sizes: tuple[int, ...] = (50,),
    ols_best_degree: int = 10,
    gamma: float = 0.01,
    epochs: int = 1000,
    batch_size: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Fit OLS, this project's FFNN and Scikit-Learn's MLP on the same data.
    Args:
        x_train: Training input data.
        y_train: Training target data.
        x_grid: Grid of input data for predictions.
        hidden_sizes: Tuple specifying the number of nodes in each hidden layer.
        ols_best_degree: Degree of the polynomial for the OLS fit.
        gamma: Learning rate for the optimizers.
        epochs: Number of training epochs.
        batch_size: Size of the mini-batches for training.
        seed: Random seed for reproducibility.

    The hidden layers use sigmoid activations, as specified in part b) of the project instructions.
    The data is scaled.
    """
    # Scale data
    X_scaled, x_mean, x_std, y_centered, y_mean = scale_data(x_train, y_train)
    X_grid_scaled = (x_grid - x_mean) / x_std

    # OLS
    ols_fitter = ModelFitter(OLS)
    _, ols_x_mean, ols_x_std, ols_y_mean = fit_polynomial(
        ols_fitter, x_train, y_train, ols_best_degree
    )
    ols_train_prediction = predict_polynomial(
        ols_fitter, x_train, ols_best_degree, ols_x_mean, ols_x_std, ols_y_mean
    )
    ols_grid_prediction = predict_polynomial(
        ols_fitter, x_grid, ols_best_degree, ols_x_mean, ols_x_std, ols_y_mean
    )

    # Our feed-forward neural network.
    layers = tuple((nodes, Sigmoid) for nodes in hidden_sizes) + ((1, Identity),)
    network_kwargs = {
        "n_inputs": X_scaled.shape[1],
        "layers": layers,
        "seed": seed,
    }
    trainer = NetworkTrainer(
        network_kwargs=network_kwargs,
        optimizer=Adam,
        optimizer_kwargs={"gamma": gamma},
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    _ = trainer.fit(X_scaled, y_centered)
    network_train_prediction = trainer.predict(X_scaled) + y_mean
    network_grid_prediction = trainer.predict(X_grid_scaled) + y_mean

    # Scikit-Learn
    sklearn_mlp = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation="logistic",
        solver="adam",
        learning_rate_init=gamma,
        batch_size=batch_size,
        alpha=0.0,
        max_iter=epochs,
        tol=0.0,
        n_iter_no_change=epochs + 1,
        early_stopping=False,
        random_state=seed,
    )
    sklearn_mlp.fit(X_scaled, y_centered)
    sklearn_train_prediction = sklearn_mlp.predict(X_scaled).reshape(-1, 1) + y_mean
    sklearn_grid_prediction = sklearn_mlp.predict(X_grid_scaled).reshape(-1, 1) + y_mean

    n_network_parameters = sum(layer.W.size + layer.b.size for layer in trainer.network.layers)
    n_mlp_parameters = sum(
        weights.size + biases.size
        for weights, biases in zip(sklearn_mlp.coefs_, sklearn_mlp.intercepts_, strict=True)
    )

    results = {
        "Analytical OLS": {
            "train_r2": float(R2(y_train, ols_train_prediction)),
            "MSE": float(MSE(y_train, ols_train_prediction)),
            "grid_prediction": ols_grid_prediction.ravel(),
            "n_parameters": ols_best_degree + 1,  # Include the intercept
        },
        "Our FFNN": {
            "train_r2": float(R2(y_train, network_train_prediction)),
            "MSE": float(MSE(y_train, network_train_prediction)),
            "grid_prediction": network_grid_prediction.ravel(),
            "n_parameters": int(n_network_parameters),
        },
        "Scikit-Learn MLP": {
            "train_r2": float(R2(y_train, sklearn_train_prediction)),
            "MSE": float(MSE(y_train, sklearn_train_prediction)),
            "grid_prediction": sklearn_grid_prediction.ravel(),
            "n_parameters": int(n_mlp_parameters),
        },
    }
    return results


def run_optimizer_sweep(
    x_train: Array,
    y_train: Array,
    gammas: Array,
    batch_sizes: dict[str, int | None],
    epochs: int = 1000,
    seed: int = 2026,
    hidden_sizes: tuple[int] = (50,),
) -> dict[str, Any]:
    """Select gamma by final training MSE and retain its learning curve."""
    X_scaled, _, _, y_centered, _ = scale_data(x_train, y_train)
    results = {}

    for name, optimizer_class in OPTIMIZERS.items():
        train_mse = np.full((len(gammas), epochs), np.nan)
        for i, gamma in enumerate(gammas):
            trainer = NetworkTrainer(
                network_kwargs={
                    "n_inputs": X_scaled.shape[1],
                    "layers": (
                        tuple((nodes, Sigmoid) for nodes in hidden_sizes) + ((1, Identity),)
                    ),
                    "seed": seed,
                },
                optimizer=optimizer_class,
                optimizer_kwargs={"gamma": float(gamma)},
                epochs=epochs,
                batch_size=batch_sizes[name],
                seed=seed,
            )
            history = trainer.fit(X_scaled, y_centered)
            train_mse[i] = 2.0 * np.asarray(
                history["cost"], dtype=float
            )  # Multiply by 2 to get the MSE from the cost (which is 0.5 * MSE)

        final_mse = train_mse[:, -1]
        scores = np.where(np.isfinite(final_mse), final_mse, np.inf)
        best_index = np.argmin(scores)

        if not np.isfinite(scores[best_index]):
            raise ValueError(f"No finite final training MSE for {name}.")

        results[name] = {
            "final_mse": final_mse,
            "best_gamma": float(gammas[best_index]),
            "train_mse": train_mse[best_index],
        }
    return results
