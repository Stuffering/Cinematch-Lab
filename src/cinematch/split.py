"""Dataset splitting utilities for MovieLens ratings."""

from __future__ import annotations

import pandas as pd


def split_ratings_by_time(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets by timestamp."""
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    sorted_ratings = ratings.sort_values(
        ["timestamp", "user_id", "item_id"]
    ).reset_index(drop=True)

    split_index = int(len(sorted_ratings) * (1 - test_size))
    if split_index <= 0 or split_index >= len(sorted_ratings):
        raise ValueError("test_size is too large or too small for the dataset size.")

    train = sorted_ratings.iloc[:split_index].reset_index(drop=True)
    test = sorted_ratings.iloc[split_index:].reset_index(drop=True)

    return train, test


def split_ratings_train_valid_test_by_time(
    ratings: pd.DataFrame,
    valid_size: float = 0.2,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into train, validation, and test sets by timestamp."""
    if valid_size <= 0 or valid_size >= 1:
        raise ValueError("valid_size must be between 0 and 1.")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")
    if valid_size + test_size >= 1:
        raise ValueError("valid_size and test_size must sum to less than 1.")

    sorted_ratings = ratings.sort_values(
        ["timestamp", "user_id", "item_id"]
    ).reset_index(drop=True)

    train_end = int(len(sorted_ratings) * (1 - valid_size - test_size))
    valid_end = int(len(sorted_ratings) * (1 - test_size))

    if train_end <= 0 or valid_end <= train_end or valid_end >= len(sorted_ratings):
        raise ValueError(
            "valid_size and test_size are too large or too small for the dataset size."
        )

    train = sorted_ratings.iloc[:train_end].reset_index(drop=True)
    valid = sorted_ratings.iloc[train_end:valid_end].reset_index(drop=True)
    test = sorted_ratings.iloc[valid_end:].reset_index(drop=True)

    return train, valid, test
