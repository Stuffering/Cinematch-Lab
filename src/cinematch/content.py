"""Content-based recommendation utilities."""

from __future__ import annotations

import pandas as pd


def get_genre_columns(movies: pd.DataFrame) -> list[str]:
    """Return the movie genre feature columns."""
    metadata_columns = {
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    }

    return [
        column
        for column in movies.columns
        if column not in metadata_columns
    ]


def build_movie_feature_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Build a movie-by-feature matrix from explicit movie metadata."""
    genre_columns = get_genre_columns(movies)

    features = movies.set_index("movie_id")[genre_columns]

    return features.sort_index()


def build_user_profile(
    ratings: pd.DataFrame,
    movie_features: pd.DataFrame,
    user_id: int,
) -> pd.Series:
    """Build one user's content preference profile from rated movies."""
    user_history = ratings.loc[ratings["user_id"] == user_id]
    if user_history.empty:
        raise ValueError(f"user_id {user_id} is not present in the ratings table.")

    user_history = user_history.loc[
        user_history["item_id"].isin(movie_features.index)
    ]
    if user_history.empty:
        return pd.Series(0.0, index=movie_features.columns, name="profile")

    rated_features = movie_features.loc[user_history["item_id"]]
    ratings_values = user_history["rating"].to_numpy(dtype=float)
    centered_weights = ratings_values - ratings_values.mean()
    normalizer = abs(centered_weights).sum()

    if normalizer == 0:
        return pd.Series(0.0, index=movie_features.columns, name="profile")

    weighted_features = rated_features.mul(centered_weights, axis=0)
    profile = weighted_features.sum(axis=0) / normalizer

    return profile.rename("profile")


def recommend_content_based(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies whose content features match a user's profile."""
    movie_features = build_movie_feature_matrix(movies)
    user_profile = build_user_profile(
        ratings=ratings,
        movie_features=movie_features,
        user_id=user_id,
    )

    scores = movie_features.dot(user_profile)

    rated_items = set(ratings.loc[ratings["user_id"] == user_id, "item_id"])
    scores = scores.drop(index=rated_items, errors="ignore")

    top_scores = scores.sort_values(ascending=False).head(n)

    movie_titles = movies.set_index("movie_id")["title"]
    recommendations = pd.DataFrame(
        {
            "movie_id": top_scores.index,
            "title": movie_titles.loc[top_scores.index].to_numpy(),
            "score": top_scores.to_numpy(),
        }
    )

    return recommendations.reset_index(drop=True)
