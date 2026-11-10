"""
Ordinary Least Squares (OLS) model implementation.

Tools used:
    - Copilot autocomplete
"""

import numpy as np

from project1.typing import Array

from .base import BaseModel


class OLS(BaseModel):
    def __init__(self, X: Array, y: Array):
        super().__init__(X, y)

    def closed_form(self) -> Array:
        """Compute the closed-form solution for the OLS model parameters, through the SVD"""
        U, s, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.theta = np.asarray(Vt.T @ np.diag(1 / s) @ U.T @ self.y, dtype=np.float64)
        return self.theta
