"""Matrix factorization rating prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MatrixFactorizationModel:
    """Predict ratings with user and item latent factors."""

    def __init__(
        self,
        n_factors: int = 10,
        learning_rate: float = 0.02,
        regularization: float = 0.1,
        n_epochs: int = 60,
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
        self.global_mean_ = float(ratings["rating"].mean())

        user_ids = sorted(ratings["user_id"].unique())
        item_ids = sorted(ratings["item_id"].unique())

        self.user_index_ = {
            int(user_id): index
            for index, user_id in enumerate(user_ids)
        }
        self.item_index_ = {
            int(item_id): index
            for index, item_id in enumerate(item_ids)
        }

        rng = np.random.default_rng(self.random_state)

        self.user_biases_ = np.zeros(len(user_ids))
        self.item_biases_ = np.zeros(len(item_ids))

        self.user_factors_ = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(len(user_ids), self.n_factors),
        )
        self.item_factors_ = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(len(item_ids), self.n_factors),
        )

        for _ in range(self.n_epochs):
            shuffled = ratings.sample(
                frac=1.0,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )

            for row in shuffled.itertuples(index=False):
                user_id = int(row.user_id)
                item_id = int(row.item_id)
                rating = float(row.rating)

                user_position = self.user_index_[user_id]
                item_position = self.item_index_[item_id]

                user_bias = self.user_biases_[user_position]
                item_bias = self.item_biases_[item_position]
                user_factors = self.user_factors_[user_position].copy()
                item_factors = self.item_factors_[item_position].copy()

                prediction = (
                    self.global_mean_
                    + user_bias
                    + item_bias
                    + float(np.dot(user_factors, item_factors))
                )
                error = rating - prediction

                self.user_biases_[user_position] += self.learning_rate * (
                    error - self.regularization * user_bias
                )
                self.item_biases_[item_position] += self.learning_rate * (
                    error - self.regularization * item_bias
                )

                self.user_factors_[user_position] += self.learning_rate * (
                    error * item_factors - self.regularization * user_factors
                )
                self.item_factors_[item_position] += self.learning_rate * (
                    error * user_factors - self.regularization * item_factors
                )

        return self

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for user-item rows."""
        if (
            self.global_mean_ is None
            or self.user_index_ is None
            or self.item_index_ is None
            or self.user_biases_ is None
            or self.item_biases_ is None
            or self.user_factors_ is None
            or self.item_factors_ is None
        ):
            raise ValueError("The model has not been fitted yet.")

        predictions: list[float] = []

        for row in ratings.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)

            prediction = self.global_mean_

            user_position = self.user_index_.get(user_id)
            item_position = self.item_index_.get(item_id)

            if user_position is not None:
                prediction += self.user_biases_[user_position]

            if item_position is not None:
                prediction += self.item_biases_[item_position]

            if user_position is not None and item_position is not None:
                prediction += float(
                    np.dot(
                        self.user_factors_[user_position],
                        self.item_factors_[item_position],
                    )
                )

            predictions.append(float(np.clip(prediction, 1.0, 5.0)))

        return pd.Series(
            predictions,
            index=ratings.index,
            name="prediction",
        )
