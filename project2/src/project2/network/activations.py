"""Activation functions for the neural network.

Tools used:
    - Copilot Autocomplete

"""

from abc import ABC, abstractmethod

import numpy as np

from project2.network.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, Loss, MSELoss
from project2.typing import Array


class Activation(ABC):
    loss_type: type[Loss] | None = (
        None  # The type of loss function associated with this activation function.
    )
    output_neurons: int | None = None

    @abstractmethod
    def forward(self, z: Array) -> Array:
        pass

    @abstractmethod
    def derivative(self, z: Array, a: Array) -> Array:
        pass


class Identity(Activation):
    loss_type = MSELoss
    output_neurons = 1

    def forward(self, z: Array) -> Array:
        return z

    def derivative(self, z: Array, _: Array) -> Array:
        return np.ones_like(z)


class Sigmoid(Activation):
    loss_type = BinaryCrossEntropyLoss
    output_neurons = 1

    def forward(self, z: Array) -> Array:
        return 1 / (1 + np.exp(-z))

    def derivative(self, _: Array, a: Array) -> Array:
        return a * (1 - a)


class Softmax(Activation):
    loss_type = CrossEntropyLoss

    def forward(self, z: Array) -> Array:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def derivative(self, _z: Array, _a: Array) -> Array:
        raise NotImplementedError("Softmax derivative is not implemented.")


class ReLU(Activation):
    def forward(self, z: Array) -> Array:
        return np.maximum(0, z)

    def derivative(self, z: Array, _: Array) -> Array:
        return (z > 0).astype(float)


class LReLU(Activation):  # TODO and what is a)
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, z: Array) -> Array:
        return np.where(z > 0, z, self.alpha * z)

    def derivative(self, z: Array, a: Array) -> Array:
        dz = np.ones_like(z)
        dz[z < 0] = self.alpha
        return dz


class Tanh(Activation):
    def forward(self, z: Array) -> Array:
        return np.tanh(z)

    def derivative(self, z: Array, a: Array) -> Array:
        return 1 - a**2
