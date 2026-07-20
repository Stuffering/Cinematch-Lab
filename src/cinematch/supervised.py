"""Supervised rating prediction utilities."""

from __future__ import annotations

import pandas as pd


def build_rating_feature_table(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Join ratings with user and movie metadata for supervised learning."""
    raise NotImplementedError("Stage 07: build supervised rating feature table")


def split_features_and_target(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature table into model inputs and rating target."""
    raise NotImplementedError("Stage 07: split features and target")


def build_preprocessing_feature_lists(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns for preprocessing."""
    raise NotImplementedError("Stage 07: identify supervised feature types")
