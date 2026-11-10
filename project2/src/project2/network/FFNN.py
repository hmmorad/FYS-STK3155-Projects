"""Feedforward Neural Network (FFNN) implementation.

Tools used:
    - Copilot Autocomplete
"""

import numpy as np

from project2.network.activations import Activation, Identity
from project2.network.layer import DenseLayer
from project2.typing import Array


class NeuralNetwork:
    def __init__(
        self,
        n_inputs: int,
        layers: tuple[tuple[int, type[Activation]], ...] = ((50, Identity),),
        lmbd1: float = 0.0,
        lmbd2: float = 0.0,
        seed: int = 2026,
    ) -> None:
        """Initialize the feedforward neural network (FFNN).

        Args:
            n_inputs (int): Number of input features.
            layers (tuple[tuple[int, type[Activation]]]): Hidden layer size and activation in tuple.
            seed (int): Random seed for weight initialization.
        """
        self.rng = np.random.default_rng(seed)
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2

        sizes = [n_inputs, *(size for size, _ in layers)]
        self.layers: list[DenseLayer] = [
            DenseLayer(
                n_inputs=sizes[i], n_neurons=sizes[i + 1], activation=activation(), rng=self.rng
            )
            for i, (_, activation) in enumerate(layers)
        ]

        # Ensure the last layer's activation has an associated loss type
        # (e.g., Softmax with CrossEntropy, Identity with MSE)
        self.last_activation = self.layers[-1].activation
        loss_type = self.last_activation.loss_type
        if loss_type is None:
            raise ValueError(
                "The last layer's output activation must have an associated loss type."
            )
        self.loss = loss_type()

        # Ensure the last layer's output neurons match the activation's expected output neurons.
        # (e.g., Identity with MSE expects 1 output neuron)
        expected = self.last_activation.output_neurons
        if expected is not None and sizes[-1] != expected:
            raise ValueError(
                "The last layer's output neurons must match the activation's expected output neurons."  # noqa
            )

    def forward(self, X: Array) -> Array:
        """Perform a forward pass through the network."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true: Array, y_pred: Array) -> None:
        """Perform a backward pass through the network."""
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Target shape {y_true.shape} does not match prediction shape {y_pred.shape}."
            )

        n = y_true.shape[0]
        # Combined output activation and loss gradient
        dA = (y_pred - y_true) / n  # Average gradient

        # Backpropagate through the layers in reverse order
        for i, layer in enumerate(reversed(self.layers)):
            dA = layer.backward(dA, self.lmbd1, self.lmbd2, is_last_layer=(i == 0))

    def cost(self, X: Array, y: Array) -> tuple[float, float]:
        """Calculate the cost of the network, and possible regularization."""
        y_pred = self.forward(X)
        if y.shape != y_pred.shape:
            raise ValueError(
                f"Target shape {y.shape} does not match prediction shape {y_pred.shape}."
            )

        data_cost = self.loss.compute(y_pred, y)
        regularization = sum(
            self.lmbd1 * np.sum(np.abs(layer.W)) + self.lmbd2 * np.sum(layer.W**2)
            for layer in self.layers
        )
        return float(data_cost), float(regularization)

    def predict(self, X: Array) -> Array:
        """Predict the output for the given input."""
        return self.forward(X)
