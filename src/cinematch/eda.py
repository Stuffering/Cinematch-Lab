"""Exploratory data analysis helpers for MovieLens tables."""
import pandas as pd

from cinematch.data import MOVIE_COLUMNS


def summarize_ratings(ratings: pd.DataFrame) -> dict[str, object]:
    """Summarize the MovieLens ratings table."""
    # TODO(student): Return basic rating table statistics.
    # HINT: Start with row count, unique users, unique items, rating min/max/mean.
    # REFERENCE: docs/stage_02_eda_cleaning.md
    rating_distribution = ratings["rating"].value_counts().sort_index().to_dict()
    ratings_per_user = ratings.groupby("user_id").size()
    ratings_per_item = ratings.groupby("item_id").size()
    duplicate_rating_events = ratings.duplicated(
        subset=["user_id", "item_id", "timestamp"]
    ).sum()
    rated_at = pd.to_datetime(ratings["timestamp"], unit="s")

    return {
        "row_count": ratings.shape[0],
        "unique_users": ratings["user_id"].nunique(),
        "unique_items": ratings["item_id"].nunique(),
        "rating_min": ratings["rating"].min(),
        "rating_max": ratings["rating"].max(),
        "rating_mean": ratings["rating"].mean(),
        "rating_median": ratings["rating"].median(),
        "rating_distribution": rating_distribution,
        "ratings_per_user_min": ratings_per_user.min(),
        "ratings_per_user_max": ratings_per_user.max(),
        "ratings_per_user_mean": ratings_per_user.mean(),
        "ratings_per_user_median": ratings_per_user.median(),
        "ratings_per_item_min": ratings_per_item.min(),
        "ratings_per_item_max": ratings_per_item.max(),
        "ratings_per_item_mean": ratings_per_item.mean(),
        "ratings_per_item_median": ratings_per_item.median(),
        "duplicate_rating_events": duplicate_rating_events,
        "rated_at_min": rated_at.min(),
        "rated_at_max": rated_at.max(),
    }


def summarize_users(users: pd.DataFrame) -> dict[str, object]:
    """Summarize the MovieLens users table."""

    gender_distribution = users["gender"].value_counts().sort_index().to_dict()
    occupation_distribution = (
        users["occupation"].value_counts().sort_index().to_dict()
    )

    return {
        "row_count": users.shape[0],
        "unique_users": users["user_id"].nunique(),
        "age_min": users["age"].min(),
        "age_max": users["age"].max(),
        "age_mean": users["age"].mean(),
        "age_median": users["age"].median(),
        "gender_distribution": gender_distribution,
        "occupation_distribution": occupation_distribution,
        "duplicate_user_ids": users["user_id"].duplicated().sum(),
    }


def summarize_movies(movies: pd.DataFrame) -> dict[str, object]:
    """Summarize the MovieLens movies table."""
    genre_columns = MOVIE_COLUMNS[5:]
    genre_count_per_movie = movies[genre_columns].sum(axis=1)
    return {
        "row_count": movies.shape[0],
        "unique_movies": movies["movie_id"].nunique(),
        "duplicate_movie_ids": movies["movie_id"].duplicated().sum(),
        "missing_titles": movies["title"].isnull().sum(),
        "genre_totals": movies[genre_columns].sum().to_dict(),
        "movies_without_genres": (genre_count_per_movie == 0).sum(),
        "movies_with_multiple_genres": (genre_count_per_movie > 1).sum(),
    }


def build_data_profile(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> dict[str, object]:
    """Build a data profile for the MovieLens dataset."""
    return {
        "ratings": summarize_ratings(ratings),
        "users": summarize_users(users),
        "movies": summarize_movies(movies),
    }