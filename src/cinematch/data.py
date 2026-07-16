"""Data loading utilities for the MovieLens dataset."""

from pathlib import Path

import pandas as pd

RATING_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
]

MOVIE_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

def load_ratings(data_directory: Path) -> pd.DataFrame:
    """Load the MovieLens ratings table."""
    ratings_path = data_directory / "u.data"

    return pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=RATING_COLUMNS,
    )

def load_users(data_directory: Path) -> pd.DataFrame:
    """Load the MovieLens users table."""
    users_path = data_directory / "u.user"

    return pd.read_csv(
        users_path,
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

def load_movies(data_directory: Path) -> pd.DataFrame:
    """Load the MovieLens movies table."""
    movies_path = data_directory / "u.item"

    return pd.read_csv(
        movies_path,
        sep="|",
        header=None,
        names=MOVIE_COLUMNS,
        encoding="latin-1",
    )

def load_dataset(
    data_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all MovieLens tables."""
    ratings = load_ratings(data_directory)
    users = load_users(data_directory)
    movies = load_movies(data_directory)

    return ratings, users, movies

def validate_dataset(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    """Validate the basic MovieLens table relationships."""
    required_rating_columns = {
        "user_id",
        "item_id",
        "rating",
        "timestamp",
    }

    required_user_columns = {
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
    }

    required_movie_columns = {
        "movie_id",
        "title",
    }

    missing_rating_columns = required_rating_columns - set(ratings.columns)
    missing_user_columns = required_user_columns - set(users.columns)
    missing_movie_columns = required_movie_columns - set(movies.columns)

    if missing_rating_columns:
        raise ValueError(
            f"Ratings table is missing columns: {missing_rating_columns}"
        )

    if missing_user_columns:
        raise ValueError(
            f"Users table is missing columns: {missing_user_columns}"
        )

    if missing_movie_columns:
        raise ValueError(
            f"Movies table is missing columns: {missing_movie_columns}"
        )

    if ratings["rating"].min() < 1 or ratings["rating"].max() > 5:
        raise ValueError(
            "Ratings table must have ratings between 1 and 5, "
            f"but has ratings between {ratings['rating'].min()} "
            f"and {ratings['rating'].max()}"
        )

    missing_user = set(ratings["user_id"]) - set(users["user_id"])
    if missing_user:
        raise ValueError(
            "Ratings table contains user_ids that are not in the users table: "
            f"{missing_user}"
        )

    missing_movie = set(ratings["item_id"]) - set(movies["movie_id"])
    if missing_movie:
        raise ValueError(
            "Ratings table contains item_ids that are not in the movies table: "
            f"{missing_movie}"
        )
