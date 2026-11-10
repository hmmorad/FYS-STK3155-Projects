"""Script for part c): bootstrap, bias-variance trade-off.

Tools used:
    - Github Copilot Autocomplete

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from project1.model_fitter import ModelFitter
from project1.models import OLS
from project1.resampling import Bootstrap

if TYPE_CHECKING:
    from project1.typing import Array


def run(
    x: Array,
    y: Array,
    max_degree: int = 15,
    n_bootstraps: int = 100,
    test_size: float = 0.2,
    seed: int = 2026,
) -> dict[str, Array | np.ndarray]:
    """Run train/test and bootstrap bias-variance analyses."""
    bootstrap = Bootstrap(
        x=x,
        y=y,
        model_fitter=ModelFitter(model_class=OLS),
        n_bootstraps=n_bootstraps,
        test_size=test_size,
        seed=seed,
    )

    error, bias2, variance = bootstrap.bias_variance_sweep(max_degree)

    return {
        "degrees": np.arange(1, max_degree + 1),
        "error": error,
        "bias2": bias2,
        "variance": variance,
    }
