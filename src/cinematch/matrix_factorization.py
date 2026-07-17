"""Matrix factorization model scaffold."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MatrixFactorizationModel:
    """Predict ratings with user and item latent factors."""

    def __init__(
        self,
        n_factors: int = 10,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        n_epochs: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.global_mean_: float | None = None
        self.user_index_: dict[int, int] | None = None
        self.item_index_: dict[int, int] | None = None
        self.user_biases_: np.ndarray | None = None
        self.item_biases_: np.ndarray | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame) -> MatrixFactorizationModel:
        """Fit latent factors from a ratings table."""
        # TODO(student): Build mappings, initialize parameters, and run SGD.
        raise NotImplementedError("Stage 05 TODO: implement matrix factorization fit")

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for user-item rows."""
        # TODO(student): Use learned biases and latent factors for each row.
        raise NotImplementedError(
            "Stage 05 TODO: implement matrix factorization predict"
        )
