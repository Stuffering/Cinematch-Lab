"""Recommend unseen movies for a user from explicit content features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.content import recommend_content_based

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def recommend_content_for_user(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 10,
) -> pd.DataFrame:
    """Return content-based movie recommendations for one user."""
    return recommend_content_based(
        ratings=ratings,
        movies=movies,
        user_id=user_id,
        n=n,
    )


def main() -> None:
    """Recommend unseen movies from prepared MovieLens content features."""
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
    args = parser.parse_args()

    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")
    movies = pd.read_csv(args.processed_directory / "movies.csv")

    recommendations = recommend_content_for_user(
        ratings=ratings,
        movies=movies,
        user_id=args.user_id,
        n=args.n,
    )

    print(f"content-based recommended movies for user {args.user_id}")
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
