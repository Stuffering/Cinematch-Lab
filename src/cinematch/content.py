"""Content-based recommendation utilities."""

from __future__ import annotations

import pandas as pd


def get_genre_columns(movies: pd.DataFrame) -> list[str]:
    """Return the movie genre feature columns."""
    raise NotImplementedError("Stage 06: identify movie genre feature columns")


def build_movie_feature_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """Build a movie-by-feature matrix from explicit movie metadata."""
    raise NotImplementedError("Stage 06: build movie genre feature matrix")


def build_user_profile(
    ratings: pd.DataFrame,
    movie_features: pd.DataFrame,
    user_id: int,
) -> pd.Series:
    """Build one user's content preference profile from rated movies."""
    raise NotImplementedError("Stage 06: build user content profile")


def recommend_content_based(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies whose content features match a user's profile."""
    raise NotImplementedError("Stage 06: recommend movies from content features")
