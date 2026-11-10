"""
Generate Runge function data.

Tools used:
    - Copilot autocomplete
"""

import numpy as np

from project1.typing import Array


def runge_function(n: int = 100, noise: float = 0.1, seed: int = 2026) -> tuple[Array, Array]:
    """Return Runge-function: 1/(1+25x^2) on [-1,1]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 1.0 / (1.0 + 25.0 * x**2) + noise * rng.standard_normal(n)
    return x, y
