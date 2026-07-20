"""Recommend movies with a hybrid blend of item-CF and content signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.collaborative import (
    build_user_item_matrix,
    center_user_ratings,
    recommend_items_for_user,
)
from cinematch.content import recommend_content_based
from cinematch.hybrid import recommend_hybrid, standardize_recommendations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def recommend_item_cf_for_user(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int = 25,
    min_rating: float = 4.0,
    min_common_users: int = 20,
) -> pd.DataFrame:
    """Return item-CF recommendations with movie titles for one user."""
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

    if scores.empty:
        return pd.DataFrame(columns=["movie_id", "title", "score"])

    recommendations = scores.reset_index()
    recommendations.columns = ["movie_id", "score"]

    return recommendations.merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left",
    )[["movie_id", "title", "score"]]


def main() -> None:
    """Run hybrid recommendations from prepared MovieLens data."""
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
        help="Number of final hybrid recommendations to print",
    )
    parser.add_argument(
        "--candidate-n",
        type=int,
        default=25,
        help="Number of candidates collected from each source before blending",
    )
    parser.add_argument(
        "--item-cf-weight",
        type=float,
        default=0.7,
        help="Weight applied to item-based collaborative filtering scores",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.3,
        help="Weight applied to content-based recommendation scores",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating treated as a liked movie for item-CF",
    )
    parser.add_argument(
        "--min-common-users",
        type=int,
        default=20,
        help="Minimum users who rated both source and candidate movies for item-CF",
    )
    args = parser.parse_args()

    candidate_n = max(args.candidate_n, args.n)
    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")
    movies = pd.read_csv(args.processed_directory / "movies.csv")

    item_cf_recommendations = recommend_item_cf_for_user(
        ratings=ratings,
        movies=movies,
        user_id=args.user_id,
        n=candidate_n,
        min_rating=args.min_rating,
        min_common_users=args.min_common_users,
    )
    content_recommendations = recommend_content_based(
        ratings=ratings,
        movies=movies,
        user_id=args.user_id,
        n=candidate_n,
    )

    recommendation_frames = [
        standardize_recommendations(
            item_cf_recommendations,
            source="item_cf",
        ),
        standardize_recommendations(
            content_recommendations,
            source="content",
        ),
    ]
    source_weights = {
        "item_cf": args.item_cf_weight,
        "content": args.content_weight,
    }

    recommendations = recommend_hybrid(
        recommendation_frames=recommendation_frames,
        source_weights=source_weights,
        n=args.n,
    )

    print(f"Hybrid recommendations for user {args.user_id}")
    print(
        "Source weights: "
        f"item_cf={args.item_cf_weight:.2f}, "
        f"content={args.content_weight:.2f}"
    )
    print(f"Candidates per source: {candidate_n}")
    print()

    if recommendations.empty:
        print("No recommendations found with the current settings.")
        return

    print(
        recommendations.to_string(
            index=False,
            formatters={"hybrid_score": "{:.3f}".format},
        )
    )


if __name__ == "__main__":
    main()
