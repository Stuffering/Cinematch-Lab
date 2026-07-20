"""Supervised rating prediction utilities."""

from __future__ import annotations

import pandas as pd


def build_rating_feature_table(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Join ratings with user and movie metadata for supervised learning."""
    feature_table = ratings.merge(
        users,
        on="user_id",
        how="left",
    )

    feature_table = feature_table.merge(
        movies,
        left_on="item_id",
        right_on="movie_id",
        how="left",
    )

    return feature_table


def split_features_and_target(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature table into model inputs and rating target."""
    target = feature_table["rating"]

    excluded_columns = [
        "rating",
        "user_id",
        "item_id",
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "zip_code",
        "timestamp",
    ]
    features = feature_table.drop(
        columns=[
            column
            for column in excluded_columns
            if column in feature_table.columns
        ]
    )

    return features, target


def add_rating_history_features(
    feature_table: pd.DataFrame,
    train_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Add user and item history features computed from training ratings."""
    user_stats = train_ratings.groupby("user_id")["rating"].agg(
        user_rating_count="size",
        user_mean_rating="mean",
    )
    item_stats = train_ratings.groupby("item_id")["rating"].agg(
        item_rating_count="size",
        item_mean_rating="mean",
    )

    enriched = feature_table.merge(
        user_stats,
        on="user_id",
        how="left",
    )
    enriched = enriched.merge(
        item_stats,
        on="item_id",
        how="left",
    )

    global_mean = train_ratings["rating"].mean()

    enriched["user_rating_count"] = enriched["user_rating_count"].fillna(0)
    enriched["item_rating_count"] = enriched["item_rating_count"].fillna(0)
    enriched["user_mean_rating"] = enriched["user_mean_rating"].fillna(global_mean)
    enriched["item_mean_rating"] = enriched["item_mean_rating"].fillna(global_mean)

    return enriched


def build_preprocessing_feature_lists(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns for preprocessing."""
    numeric_features = []
    categorical_features = []

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features
