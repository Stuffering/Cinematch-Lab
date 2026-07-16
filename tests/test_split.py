import pandas as pd
import pytest

from cinematch.split import (
    split_ratings_by_time,
    split_ratings_train_valid_test_by_time,
)


def test_split_ratings_by_time_creates_chronological_train_test_sets() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [2, 1, 3, 1, 2],
            "item_id": [20, 10, 30, 40, 50],
            "rating": [4, 5, 3, 2, 1],
            "timestamp": [300, 100, 500, 200, 400],
        }
    )

    train, test = split_ratings_by_time(ratings, test_size=0.4)

    assert train["timestamp"].tolist() == [100, 200, 300]
    assert test["timestamp"].tolist() == [400, 500]
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert train.index.tolist() == [0, 1, 2]
    assert test.index.tolist() == [0, 1]


def test_split_ratings_by_time_rejects_invalid_test_size() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [5, 4],
            "timestamp": [100, 200],
        }
    )

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        split_ratings_by_time(ratings, test_size=0)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        split_ratings_by_time(ratings, test_size=1)


def test_split_ratings_by_time_rejects_empty_train_or_test() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
            "timestamp": [100],
        }
    )

    with pytest.raises(ValueError, match="dataset size"):
        split_ratings_by_time(ratings, test_size=0.2)


def test_split_ratings_train_valid_test_by_time_creates_chronological_sets() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [4, 1, 6, 2, 5, 3],
            "item_id": [40, 10, 60, 20, 50, 30],
            "rating": [4, 5, 1, 3, 2, 4],
            "timestamp": [400, 100, 600, 200, 500, 300],
        }
    )

    train, valid, test = split_ratings_train_valid_test_by_time(
        ratings,
        valid_size=0.2,
        test_size=0.2,
    )

    assert train["timestamp"].tolist() == [100, 200, 300]
    assert valid["timestamp"].tolist() == [400]
    assert test["timestamp"].tolist() == [500, 600]
    assert train["timestamp"].max() <= valid["timestamp"].min()
    assert valid["timestamp"].max() <= test["timestamp"].min()
    assert train.index.tolist() == [0, 1, 2]
    assert valid.index.tolist() == [0]
    assert test.index.tolist() == [0, 1]


def test_split_ratings_train_valid_test_by_time_rejects_invalid_sizes() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "rating": [5, 4, 3],
            "timestamp": [100, 200, 300],
        }
    )

    with pytest.raises(ValueError, match="valid_size must be between 0 and 1"):
        split_ratings_train_valid_test_by_time(ratings, valid_size=0)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        split_ratings_train_valid_test_by_time(ratings, test_size=1)

    with pytest.raises(ValueError, match="sum to less than 1"):
        split_ratings_train_valid_test_by_time(
            ratings,
            valid_size=0.5,
            test_size=0.5,
        )


def test_split_ratings_train_valid_test_by_time_rejects_empty_splits() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [5, 4],
            "timestamp": [100, 200],
        }
    )

    with pytest.raises(ValueError, match="dataset size"):
        split_ratings_train_valid_test_by_time(
            ratings,
            valid_size=0.1,
            test_size=0.2,
        )
