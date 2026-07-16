import pandas as pd

from cinematch.data import MOVIE_COLUMNS
from cinematch.eda import (
    build_data_profile,
    summarize_movies,
    summarize_ratings,
    summarize_users,
)


def test_summarize_ratings_returns_basic_statistics() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "item_id": [10, 20, 10, 30],
            "rating": [5, 4, 3, 2],
            "timestamp": [100, 200, 300, 400],
        }
    )

    summary = summarize_ratings(ratings)

    assert summary["row_count"] == 4
    assert summary["unique_users"] == 3
    assert summary["unique_items"] == 3
    assert summary["rating_min"] == 2
    assert summary["rating_max"] == 5
    assert summary["rating_mean"] == 3.5
    assert summary["rating_median"] == 3.5
    assert summary["rating_distribution"] == {
        2: 1,
        3: 1,
        4: 1,
        5: 1,
    }
    assert summary["ratings_per_user_min"] == 1
    assert summary["ratings_per_user_max"] == 2
    assert summary["ratings_per_user_mean"] == 1.3333333333333333
    assert summary["ratings_per_user_median"] == 1.0
    assert summary["ratings_per_item_min"] == 1
    assert summary["ratings_per_item_max"] == 2
    assert summary["ratings_per_item_mean"] == 1.3333333333333333
    assert summary["ratings_per_item_median"] == 1.0
    assert summary["duplicate_rating_events"] == 0
    assert summary["rated_at_min"] == pd.Timestamp("1970-01-01 00:01:40")
    assert summary["rated_at_max"] == pd.Timestamp("1970-01-01 00:06:40")


def test_summarize_users_returns_basic_statistics() -> None:
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 3],
            "age": [20, 30, 40, 40],
            "gender": ["M", "F", "M", "M"],
            "occupation": ["student", "engineer", "student", "student"],
            "zip_code": ["10001", "10002", "10003", "10003"],
        }
    )

    summary = summarize_users(users)

    assert summary["row_count"] == 4
    assert summary["unique_users"] == 3
    assert summary["age_min"] == 20
    assert summary["age_max"] == 40
    assert summary["age_mean"] == 32.5
    assert summary["age_median"] == 35.0
    assert summary["gender_distribution"] == {"F": 1, "M": 3}
    assert summary["occupation_distribution"] == {
        "engineer": 1,
        "student": 3,
    }
    assert summary["duplicate_user_ids"] == 1


def test_summarize_movies_returns_basic_statistics() -> None:
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 20, 30],
            "title": ["Movie A", "Movie B", None, "Movie D"],
        }
    )

    for genre in MOVIE_COLUMNS[5:]:
        movies[genre] = 0
    
    movies["Action"] = [1, 0, 1, 0]
    movies["Comedy"] = [0, 1, 1, 0]

    summary = summarize_movies(movies)

    assert summary["row_count"] == 4
    assert summary["unique_movies"] == 3
    assert summary["duplicate_movie_ids"] == 1
    assert summary["missing_titles"] == 1
    assert summary["genre_totals"]["Action"] == 2
    assert summary["genre_totals"]["Comedy"] == 2
    assert summary["genre_totals"]["Western"] == 0
    assert summary["movies_without_genres"] == 1
    assert summary["movies_with_multiple_genres"] == 1


def test_build_data_profile_combines_table_summaries() -> None:
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
            "age": [20, 30],
            "gender": ["M", "F"],
            "occupation": ["student", "engineer"],
            "zip_code": ["10001", "10002"],
        }
    )

    movies = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Movie A", "Movie B"],
        }
    )

    for genre in MOVIE_COLUMNS[5:]:
        movies[genre] = 0

    movies["Action"] = [1, 0]

    profile = build_data_profile(ratings, users, movies)

    assert set(profile.keys()) == {"ratings", "users", "movies"}
    assert profile["ratings"]["row_count"] == 3
    assert profile["users"]["unique_users"] == 2
    assert profile["movies"]["genre_totals"]["Action"] == 1