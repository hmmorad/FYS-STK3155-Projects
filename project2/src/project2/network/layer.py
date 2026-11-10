"""Dense layer implementation for neural networks.

Tools used:
    - Copilot Autocomplete

"""

import numpy as np

from project2.network.activations import Activation, LReLU, ReLU
from project2.typing import Array


class DenseLayer:
    def __init__(
        self, n_inputs: int, n_neurons: int, activation: Activation, rng: np.random.Generator
    ):
        self.activation = activation

        # He initialization for ReLU, and Xavier-style for the others
        if isinstance(activation, (ReLU, LReLU)):
            scale = np.sqrt(2 / n_inputs)
        else:
            scale = np.sqrt(2 / (n_inputs + n_neurons))

        self.W = rng.normal(0, scale, (n_inputs, n_neurons))  # Weight matrix
        self.b = np.full(n_neurons, 0.01)  # Bias vector

        # Cached values for backpropagation
        self.input: Array | None = None
        self.z: Array | None = None
        self.output: Array | None = None

        # Gradients
        self.dW: Array | None = None
        self.db: Array | None = None

    def forward(self, X: Array) -> Array:
        """Forward pass through the dense layer."""
        self.input = X
        self.z = X @ self.W + self.b
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(
        self, dA: Array, lmbd1: float = 0.0, lmbd2: float = 0.0, is_last_layer: bool = False
    ) -> Array:
        """Backward pass through the dense layer."""
        if self.input is None or self.z is None or self.output is None:
            raise RuntimeError("backward() called before forward()")

        # Gradient with respect to activation
        dZ = dA if is_last_layer else dA * self.activation.derivative(self.z, self.output)

        # Gradients with respect to the weights and biases
        self.dW = self.input.T @ dZ
        self.db = np.sum(dZ, axis=0)

        # Regularization
        if lmbd1:
            self.dW += lmbd1 * np.sign(self.W)
        if lmbd2:
            self.dW += 2 * lmbd2 * self.W

        # Gradient passed to the previous layer
        return dZ @ self.W.T
