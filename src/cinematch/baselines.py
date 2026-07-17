"""Simple rating prediction baselines."""

from __future__ import annotations

import pandas as pd


class BiasBaseline:
    """Predict ratings with global mean plus user and item bias terms."""

    def __init__(self, regularization: float = 0.1, n_epochs: int = 10) -> None:
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.global_mean_: float | None = None
        self.user_biases_: pd.Series | None = None
        self.item_biases_: pd.Series | None = None

    def fit(self, ratings: pd.DataFrame) -> BiasBaseline:
        """Fit regularized user and item biases from a ratings table."""
        self.global_mean_ = float(ratings["rating"].mean())
        user_biases = pd.Series(0.0, index=ratings["user_id"].unique())
        item_biases = pd.Series(0.0, index=ratings["item_id"].unique())

        user_counts = ratings.groupby("user_id")["rating"].size()
        item_counts = ratings.groupby("item_id")["rating"].size()

        for _ in range(self.n_epochs):
            item_offsets = ratings["item_id"].map(item_biases).fillna(0.0)
            user_residuals = ratings["rating"] - self.global_mean_ - item_offsets
            user_sums = user_residuals.groupby(ratings["user_id"]).sum()
            user_biases = user_sums / (user_counts + self.regularization)

            user_offsets = ratings["user_id"].map(user_biases).fillna(0.0)
            item_residuals = ratings["rating"] - self.global_mean_ - user_offsets
            item_sums = item_residuals.groupby(ratings["item_id"]).sum()
            item_biases = item_sums / (item_counts + self.regularization)

        self.user_biases_ = user_biases
        self.item_biases_ = item_biases
        return self

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for the provided rows."""
        if (
            self.global_mean_ is None
            or self.user_biases_ is None
            or self.item_biases_ is None
        ):
            raise ValueError("The model has not been fitted yet.")

        user_offsets = ratings["user_id"].map(self.user_biases_).fillna(0.0)
        item_offsets = ratings["item_id"].map(self.item_biases_).fillna(0.0)
        predictions = self.global_mean_ + user_offsets + item_offsets

        return pd.Series(
            predictions,
            index=ratings.index,
            name="prediction",
        )


class GlobalMeanBaseline:
    """Predict every rating with the training-set global mean."""

    def __init__(self) -> None:
        self.global_mean_: float | None = None

    def fit(self, ratings: pd.DataFrame) -> GlobalMeanBaseline:
        """Fit the baseline from a ratings table."""
        self.global_mean_ = float(ratings["rating"].mean())
        return self

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for the provided rows."""
        if self.global_mean_ is None:
            raise ValueError("The model has not been fitted yet.")

        return pd.Series(
            self.global_mean_,
            index=ratings.index,
            name="prediction",
        )


class UserMeanBaseline:
    """Predict ratings with each user's training-set mean."""

    def __init__(self) -> None:
        self.global_mean_: float | None = None
        self.user_means_: pd.Series | None = None

    def fit(self, ratings: pd.DataFrame) -> UserMeanBaseline:
        """Fit user means from a ratings table."""
        self.global_mean_ = float(ratings["rating"].mean())
        self.user_means_ = ratings.groupby("user_id")["rating"].mean()
        return self

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for the provided rows."""
        if self.global_mean_ is None or self.user_means_ is None:
            raise ValueError("The model has not been fitted yet.")

        predictions = ratings["user_id"].map(self.user_means_).fillna(
            self.global_mean_
        )

        return pd.Series(
            predictions,
            index=ratings.index,
            name="prediction",
        )


class ItemMeanBaseline:
    """Predict ratings with each movie's training-set mean."""

    def __init__(self) -> None:
        self.global_mean_: float | None = None
        self.item_means_: pd.Series | None = None

    def fit(self, ratings: pd.DataFrame) -> ItemMeanBaseline:
        """Fit item means from a ratings table."""
        self.global_mean_ = float(ratings["rating"].mean())
        self.item_means_ = ratings.groupby("item_id")["rating"].mean()
        return self

    def predict(self, ratings: pd.DataFrame) -> pd.Series:
        """Predict ratings for the provided rows."""
        if self.global_mean_ is None or self.item_means_ is None:
            raise ValueError("The model has not been fitted yet.")

        predictions = ratings["item_id"].map(self.item_means_).fillna(
            self.global_mean_
        )

        return pd.Series(
            predictions,
            index=ratings.index,
            name="prediction",
        )
