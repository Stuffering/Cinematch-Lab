"""Recommend unseen movies for a user with item-based collaborative filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.collaborative import (
    build_user_item_matrix,
    center_user_ratings,
    recommend_items_for_user,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def recommend_movies_for_user(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 10,
    min_rating: float = 4.0,
    min_common_users: int = 20,
) -> pd.DataFrame:
    """Return movie recommendations with titles for one user."""
    user_item_matrix = build_user_item_matrix(ratings)
    centered_matrix = center_user_ratings(user_item_matrix)

    scores = recommend_items_for_user(
        ratings=ratings,
        centered_matrix=centered_matrix,
        user_id=user_id,
        n=n,
        min_rating=min_rating,
        min_common_users=min_common_users,
    )

    recommendations = scores.reset_index()
    recommendations.columns = ["movie_id", "score"]

    return recommendations.merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left",
    )[["movie_id", "title", "score"]]


def main() -> None:
    """Recommend unseen movies from prepared MovieLens data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory containing prepared MovieLens CSV files",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="MovieLens user ID used as the recommendation target",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of recommended movies to print",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating treated as a liked movie",
    )
    parser.add_argument(
        "--min-common-users",
        type=int,
        default=20,
        help="Minimum users who rated both source and candidate movies",
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")
    movies = pd.read_csv(args.processed_directory / "movies.csv")

    recommendations = recommend_movies_for_user(
        ratings=ratings,
        movies=movies,
        user_id=args.user_id,
        n=args.n,
        min_rating=args.min_rating,
        min_common_users=args.min_common_users,
    )

    print(f"recommended movies for user {args.user_id}")
    if recommendations.empty:
        print("No recommendations found with the current settings.")
        return

    print(
        recommendations.to_string(
            index=False,
            formatters={"score": "{:.3f}".format},
        )
    )


if __name__ == "__main__":
    main()
