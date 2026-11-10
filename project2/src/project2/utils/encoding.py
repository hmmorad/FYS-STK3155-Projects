"""Target encoding for classification.

Tools used:
    - Copilot Autocomplete

"""

import numpy as np

from project2.typing import Array


def one_hot(y: Array, n_classes: int = 10) -> Array:
    """Encode integer class labels as float32 indicator rows."""
    encoded: np.ndarray = np.eye(n_classes, dtype=np.float32)[np.asarray(y, dtype=int).ravel()]
    return encoded
