from pathlib import Path

import pandas as pd
import pytest

from cinematch.data import load_dataset, validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "ml-100k"

def test_load_dataset_shapes() -> None:
    ratings, users, movies = load_dataset(DATA_DIRECTORY)

    assert ratings.shape == (100000, 4)
    assert users.shape == (943, 5)
    assert movies.shape == (1682, 24)

def make_valid_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 4, 3],
            "timestamp": [100, 200, 300],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [1, 2],
            "age": [24, 30],
            "gender": ["M", "F"],
            "occupation": ["student", "engineer"],
            "zip_code": ["10001", "10002"],
        }
    )

    movies = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Movie A", "Movie B"],
            "Action": [1, 0],
        }
    )

    return ratings, users, movies

def test_validate_dataset_accepts_valid_tables() -> None:
    ratings, users, movies = make_valid_tables()

    validate_dataset(ratings, users, movies)

def test_validate_dataset_rejects_rating_out_of_range() -> None:
    ratings, users, movies = make_valid_tables()
    ratings.loc[0, "rating"] = 6  # Invalid rating

    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_dataset(ratings, users, movies)

def test_validate_dataset_rejects_missing_rating_columns() -> None:
    ratings, users, movies = make_valid_tables()
    ratings = ratings.drop(columns=["timestamp"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(ratings, users, movies)


def test_validate_dataset_rejects_unknown_user_ids() -> None:
    ratings, users, movies = make_valid_tables()
    ratings.loc[0, "user_id"] = 999

    with pytest.raises(ValueError, match="user_ids"):
        validate_dataset(ratings, users, movies)


def test_validate_dataset_rejects_unknown_item_ids() -> None:
    ratings, users, movies = make_valid_tables()
    ratings.loc[0, "item_id"] = 999

    with pytest.raises(ValueError, match="item_ids"):
        validate_dataset(ratings, users, movies)