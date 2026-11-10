"""Test backpropagation gradients against JAX.

LLM-assisted
------------
Tool: ChatGPT (November 2026)
Role: Wrote the test
Modifications: Small changes to fit our implementation, fix typing
Prompt: Gave it the code for our network, and jax gradient tests from week42tuesday.ipynb from https://github.com/EducationalMaterialUiO/MachineLearningUiO/tree/main

"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from project2.network.activations import Activation, Identity, LReLU, ReLU, Sigmoid
from project2.network.FFNN import NeuralNetwork

jax.config.update("jax_enable_x64", True)  # type: ignore


@pytest.mark.parametrize("case", ["sigmoid", "relu", "leaky_relu"])
def test_backpropagation(case: str) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 2))
    y = rng.normal(size=(5, 1))
    layers: tuple[tuple[int, type[Activation]], ...]

    if case == "sigmoid":
        layers = ((1, Sigmoid),)  # Logistic regression: no hidden layers
        y = np.array([[0.0], [1.0], [1.0], [0.0], [1.0]])
    else:
        activation = ReLU if case == "relu" else LReLU
        layers = ((3, activation), (2, activation), (1, Identity))

    net = NeuralNetwork(n_inputs=2, layers=layers, seed=42)
    net.backward(y, net.forward(X))

    params = [(jnp.array(layer.W), jnp.array(layer.b)) for layer in net.layers]

    # Independent JAX forward pass and loss.
    def cost(params: Any) -> Any:
        a = jnp.array(X)

        for W, b in params[:-1]:
            z = a @ W + b
            if case == "relu":
                a = jax.nn.relu(z)
            else:
                a = jnp.where(z > 0, z, 0.01 * z)

        W, b = params[-1]
        z = a @ W + b

        if case == "sigmoid":
            p = jnp.clip(jax.nn.sigmoid(z), 1e-12, 1 - 1e-12)
            return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))

        return 0.5 * jnp.sum((z - y) ** 2) / len(y)

    jax_gradients = jax.grad(cost)(params)

    max_weight_error = 0.0
    max_bias_error = 0.0

    for i, (layer, (dW, db)) in enumerate(zip(net.layers, jax_gradients, strict=True)):
        assert layer.dW is not None
        assert layer.db is not None

        max_weight_error = max(max_weight_error, float(np.max(np.abs(layer.dW - dW))))
        max_bias_error = max(max_bias_error, float(np.max(np.abs(layer.db - db))))

        np.testing.assert_allclose(
            layer.dW,
            dW,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"{case}, layer {i}: incorrect weight gradient",
        )
        np.testing.assert_allclose(
            layer.db,
            db,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"{case}, layer {i}: incorrect bias gradient",
        )

    architecture = " -> ".join(
        [str(X.shape[1])]
        + [f"{layer.W.shape[1]} ({type(layer.activation).__name__})" for layer in net.layers]
    )

    print(
        f"{case.capitalize()}:\nArchitecture {architecture}\n"
        f"  Gradients match JAX | max absolute difference: "
        f"weights={max_weight_error:.1e}, biases={max_bias_error:.1e}"
    )
