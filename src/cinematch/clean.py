"""Cleaning utilities for MovieLens tables."""

from __future__ import annotations

import pandas as pd


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Clean the MovieLens ratings table for downstream modeling."""
    cleaned = ratings.copy()
    cleaned = cleaned.drop_duplicates(
        subset=["user_id", "item_id", "rating", "timestamp"]
    )
    cleaned["rated_at"] = pd.to_datetime(cleaned["timestamp"], unit="s")
    cleaned = cleaned.sort_values(["timestamp", "user_id", "item_id"]).reset_index(
        drop=True
    )
    return cleaned


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Clean the MovieLens users table for downstream modeling."""
    cleaned = users.copy()
    cleaned = cleaned.drop_duplicates(subset=["user_id"])
    cleaned = cleaned.sort_values("user_id").reset_index(drop=True)
    return cleaned


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean the MovieLens movies table for downstream modeling."""
    cleaned = movies.copy()
    cleaned = cleaned.drop_duplicates(subset=["movie_id"])
    cleaned = cleaned.sort_values("movie_id").reset_index(drop=True)
    return cleaned


def clean_dataset(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean all MovieLens tables."""
    cleaned_ratings = clean_ratings(ratings)
    cleaned_users = clean_users(users)
    cleaned_movies = clean_movies(movies)

    return cleaned_ratings, cleaned_users, cleaned_movies
