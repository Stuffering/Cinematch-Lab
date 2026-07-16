import pandas as pd

from cinematch.clean import (
    clean_dataset,
    clean_movies,
    clean_ratings,
    clean_users,
)


def test_clean_ratings_removes_duplicates_and_adds_rated_at() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [2, 1, 1, 1],
            "item_id": [20, 10, 10, 30],
            "rating": [4, 5, 5, 3],
            "timestamp": [300, 100, 100, 200],
        }
    )

    cleaned = clean_ratings(ratings)

    assert "rated_at" not in ratings.columns
    assert cleaned.shape[0] == 3
    assert "rated_at" in cleaned.columns
    assert cleaned.loc[0, "rated_at"] == pd.Timestamp("1970-01-01 00:01:40")
    assert cleaned["timestamp"].tolist() == [100, 200, 300]
    assert cleaned["user_id"].tolist() == [1, 1, 2]
    assert cleaned["item_id"].tolist() == [10, 30, 20]
    assert cleaned.index.tolist() == [0, 1, 2]


def test_clean_users_removes_duplicate_users_and_sorts() -> None:
    users = pd.DataFrame(
        {
            "user_id": [3, 1, 1, 2],
            "age": [40, 20, 20, 30],
            "gender": ["M", "F", "F", "M"],
            "occupation": ["writer", "student", "student", "engineer"],
            "zip_code": ["10003", "10001", "10001", "10002"],
        }
    )

    cleaned = clean_users(users)

    assert cleaned.shape[0] == 3
    assert cleaned["user_id"].tolist() == [1, 2, 3]
    assert cleaned.index.tolist() == [0, 1, 2]
    assert users.index.tolist() == [0, 1, 2, 3]


def test_clean_movies_removes_duplicate_movies_and_sorts() -> None:
    movies = pd.DataFrame(
        {
            "movie_id": [30, 10, 10, 20],
            "title": ["Movie C", "Movie A", "Movie A duplicate", "Movie B"],
        }
    )

    cleaned = clean_movies(movies)

    assert cleaned.shape[0] == 3
    assert cleaned["movie_id"].tolist() == [10, 20, 30]
    assert cleaned["title"].tolist() == ["Movie A", "Movie B", "Movie C"]
    assert cleaned.index.tolist() == [0, 1, 2]
    assert movies.index.tolist() == [0, 1, 2, 3]


def test_clean_dataset_combines_table_cleaners() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [2, 1, 1],
            "item_id": [20, 10, 10],
            "rating": [4, 5, 5],
            "timestamp": [300, 100, 100],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [2, 1, 1],
            "age": [30, 20, 20],
            "gender": ["M", "F", "F"],
            "occupation": ["engineer", "student", "student"],
            "zip_code": ["10002", "10001", "10001"],
        }
    )

    movies = pd.DataFrame(
        {
            "movie_id": [20, 10, 10],
            "title": ["Movie B", "Movie A", "Movie A duplicate"],
        }
    )

    cleaned_ratings, cleaned_users, cleaned_movies = clean_dataset(
        ratings,
        users,
        movies,
    )

    assert cleaned_ratings.shape[0] == 2
    assert "rated_at" in cleaned_ratings.columns
    assert cleaned_users["user_id"].tolist() == [1, 2]
    assert cleaned_movies["movie_id"].tolist() == [10, 20]