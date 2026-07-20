import pandas as pd

from cinematch.content import (
    build_movie_feature_matrix,
    build_user_profile,
    get_genre_columns,
    recommend_content_based,
)


def test_get_genre_columns_excludes_movie_metadata() -> None:
    movies = pd.DataFrame(
        {
            "movie_id": [10],
            "title": ["Example Movie"],
            "release_date": ["01-Jan-1997"],
            "Action": [1],
            "Comedy": [0],
        }
    )

    assert get_genre_columns(movies) == ["Action", "Comedy"]


def test_build_movie_feature_matrix_indexes_by_movie_id() -> None:
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Action Movie", "Comedy Movie"],
            "Action": [1, 0],
            "Comedy": [0, 1],
        }
    )

    features = build_movie_feature_matrix(movies)

    assert features.index.tolist() == [10, 20]
    assert features.columns.tolist() == ["Action", "Comedy"]


def test_build_user_profile_weights_features_by_rating() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "rating": [5, 1],
        }
    )
    movie_features = pd.DataFrame(
        {
            "Action": [1, 0],
            "Comedy": [0, 1],
        },
        index=[10, 20],
    )

    profile = build_user_profile(ratings, movie_features, user_id=1)

    assert profile["Action"] > 0
    assert profile["Comedy"] < 0


def test_build_user_profile_returns_zero_without_preference_signal() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "rating": [4, 4],
        }
    )
    movie_features = pd.DataFrame(
        {
            "Action": [1, 0],
            "Comedy": [0, 1],
        },
        index=[10, 20],
    )

    profile = build_user_profile(ratings, movie_features, user_id=1)

    assert profile.tolist() == [0.0, 0.0]


def test_recommend_content_based_excludes_already_rated_movies() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Seen Action", "Unseen Action"],
            "Action": [1, 1],
            "Comedy": [0, 0],
        }
    )

    recommendations = recommend_content_based(ratings, movies, user_id=1, n=1)

    assert recommendations["movie_id"].tolist() == [20]
