"""
Ridge regression model implementation.

Tools used:
    - Copilot autocomplete
"""

import numpy as np

from project1.models.base import BaseModel
from project1.typing import Array


class Ridge(BaseModel):
    def __init__(self, X: Array, y: Array, lam: float):
        super().__init__(X, y, lam)

    def closed_form(self) -> Array:
        """Compute closed-form solution for Ridge, using the SVD"""
        n = self.X.shape[0]
        U, s, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.theta = np.asarray(
            Vt.T @ np.diag(s / (s**2 + n * self.lam)) @ U.T @ self.y, dtype=np.float64
        )
        return self.theta
