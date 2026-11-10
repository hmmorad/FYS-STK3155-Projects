"""
Lasso regression model implementation.

Tools used:
    - Copilot autocomplete
"""

from project1.models.base import BaseModel
from project1.typing import Array


class Lasso(BaseModel):
    def __init__(self, X: Array, y: Array, lam: float):
        super().__init__(X, y, lam, penalty="L_1")

    def hessian_eigs(self) -> Array:
        """Compute the eigenvalues of the Hessian matrix of the cost function for the model."""
        raise NotImplementedError("Hessian eigenvalues are not available for Lasso regression.")
