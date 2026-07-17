"""Find movies with similar user rating patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.collaborative import (
    build_user_item_matrix,
    center_user_ratings,
    cosine_similarity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def find_similar_movies(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    item_id: int,
    n: int = 10,
    min_common_users: int = 20,
) -> pd.DataFrame:
    """Return movies most similar to one movie by centered rating patterns."""
    user_item_matrix = build_user_item_matrix(ratings)
    centered_matrix = center_user_ratings(user_item_matrix)

    if item_id not in centered_matrix.columns:
        raise ValueError(f"item_id {item_id} is not present in the ratings table.")

    target_vector = centered_matrix[item_id]
    rows: list[dict[str, float | int]] = []

    for candidate_id in centered_matrix.columns:
        if candidate_id == item_id:
            continue

        candidate_vector = centered_matrix[candidate_id]
        aligned = pd.concat([target_vector, candidate_vector], axis=1).dropna()
        common_users = len(aligned)

        if common_users < min_common_users:
            continue

        rows.append(
            {
                "movie_id": int(candidate_id),
                "similarity": cosine_similarity(target_vector, candidate_vector),
                "common_users": common_users,
            }
        )

    if rows:
        similar_movies = (
            pd.DataFrame(rows)
            .sort_values(
                ["similarity", "common_users", "movie_id"],
                ascending=[False, False, True],
            )
            .head(n)
            .reset_index(drop=True)
        )
    else:
        similar_movies = pd.DataFrame(
            columns=["movie_id", "similarity", "common_users"]
        )

    movie_titles = movies[["movie_id", "title"]]

    return similar_movies.merge(movie_titles, on="movie_id", how="left")[
        ["movie_id", "title", "similarity", "common_users"]
    ]


def main() -> None:
    """Find similar movies from prepared MovieLens data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory containing prepared MovieLens CSV files",
    )
    parser.add_argument(
        "--item-id",
        type=int,
        default=50,
        help="MovieLens item ID used as the query movie",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of similar movies to print",
    )
    parser.add_argument(
        "--min-common-users",
        type=int,
        default=20,
        help="Minimum users who rated both the query and candidate movies",
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")
    movies = pd.read_csv(args.processed_directory / "movies.csv")

    query_title = movies.loc[movies["movie_id"] == args.item_id, "title"]
    query_label = (
        query_title.iloc[0] if not query_title.empty else f"movie {args.item_id}"
    )

    results = find_similar_movies(
        ratings=ratings,
        movies=movies,
        item_id=args.item_id,
        n=args.n,
        min_common_users=args.min_common_users,
    )

    print(f"query movie: {query_label} ({args.item_id})")
    if results.empty:
        print("No similar movies found with the current minimum overlap.")
        return

    print(
        results.to_string(
            index=False,
            formatters={"similarity": "{:.3f}".format},
        )
    )


if __name__ == "__main__":
    main()
