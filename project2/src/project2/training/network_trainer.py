from typing import Any

import numpy as np
from project2.network.activations import Sigmoid, Softmax
from project2.network.FFNN import NeuralNetwork
from project2.optimizers.base import Optimizer
from project2.typing import Array
from project2.utils.metrics import accuracy


class NetworkTrainer:
    def __init__(
        self,
        network_kwargs: dict[str, Any],
        optimizer: type[Optimizer],
        optimizer_kwargs: dict[str, Any] | None = None,
        epochs: int = 100,
        batch_size: int | None = None,
        gamma_step_decay: tuple[float, float] | None = None,
        seed: int = 2026,
        evaluation_batch_size: int = 1000,
    ) -> None:
        """Initialize the network trainer with network and optimizer settings.

        Args:
            network_kwargs: Arguments for the neural network (FFNN).
            optimizer: Optimizer class to use. Example: Adam, RMSprop, etc.
            optimizer_kwargs: Arguments for the optimizer.
            epochs: Number of training epochs.
            batch_size: Size of mini-batches. If None; the entire dataset is used as a single batch.
            gamma_step_decay: Tuple (t0, t1) for learning rate schedule. Inverse time decay is used.
            seed: Random seed for reproducibility.
            evaluation_batch_size: Batch size to use during evaluation on the validation set.
        """
        self.network = NeuralNetwork(**network_kwargs)
        self.optimizer = optimizer(**(optimizer_kwargs or {}))
        self.epochs = epochs
        self.batch_size: int | None = batch_size
        self.rng = np.random.default_rng(seed)
        self.gamma_step_decay = gamma_step_decay
        self.evaluation_batch_size = evaluation_batch_size
        self.is_classification = isinstance(self.network.last_activation, (Sigmoid, Softmax))

    def fit(
        self, X: Array, y: Array, X_val: Array | None = None, y_val: Array | None = None
    ) -> dict[str, list[float]]:
        """Train the neural network on the given dataset.

        If validation data is provided, it will also evaluate the validation loss and accuracy.

        Args:
            X: Training input data.
            y: Training target data.
            X_val: Validation input data.
            y_val: Validation target data.
        """
        n = X.shape[0]
        batch_size = n if self.batch_size is None else self.batch_size
        history: dict[str, list[float]] = {"cost": []}

        if self.is_classification:
            history["train_accuracy"] = []

        if X_val is not None:
            history["val_loss"] = []
            if self.is_classification:
                history["val_accuracy"] = []

        self.optimizer.reset()
        t = 0
        for _ in range(self.epochs):
            batches = self.make_batches(len(X), batch_size, self.rng)
            for batch in batches:
                t += 1
                # Update learning rate if using a learning rate schedule
                if self.gamma_step_decay is not None:
                    t0, t1 = self.gamma_step_decay
                    gamma = self.step_length(t, t0, t1)
                    self.optimizer.set_learning_rate(gamma)

                # Forward and backward pass
                y_pred = self.network.forward(X[batch])
                self.network.backward(y[batch], y_pred)
                # Update parameters
                self.optimizer.step(self.network.layers)

            # Calculate cost on the training set (including possible regularization)
            data_cost, penalty = self.network.cost(X, y)
            history["cost"].append(float(data_cost + penalty))

            if self.is_classification:
                train_prediction = self.network.forward(X)
                history["train_accuracy"].append(
                    accuracy(train_prediction, y, last_activation=self.network.last_activation)
                )

            # Evaluate test set (exclude regularization)
            if X_val is not None and y_val is not None:
                y_pred = self.network.forward(X_val)

                val_loss = self.network.loss.compute(y_pred, y_val)
                history["val_loss"].append(val_loss)
                val_accuracy = accuracy(y_pred, y_val, last_activation=self.network.last_activation)
                history["val_accuracy"].append(val_accuracy)

        return history

    def predict(self, X: Array) -> Array:
        """Predict the output for the given input data."""
        return self.network.forward(X)

    @staticmethod
    def make_batches(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Create mini-batches from the dataset."""
        idx = rng.permutation(n)
        return [idx[i : i + batch_size] for i in range(0, n, batch_size)]

    @staticmethod
    def step_length(t: int, t0: float, t1: float) -> float:
        """Compute the step length for the learning rate schedule. Inverse time decay."""
        return t0 / (t + t1)
