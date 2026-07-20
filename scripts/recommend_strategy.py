"""Recommend movies through a lightweight recommendation strategy router."""

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
from cinematch.recommendation_strategy import (
    RecommendationRequest,
    build_recommendation_request,
    choose_recommendation_mode,
    list_recommendation_modes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def count_user_ratings(ratings: pd.DataFrame, user_id: int) -> int:
    """Count how many training ratings are available for one user."""
    return int((ratings["user_id"] == user_id).sum())


def build_strategy_request_from_context(
    user_id: int,
    requested_mode: str,
    n: int,
    user_rating_count: int,
) -> tuple[RecommendationRequest, str]:
    """Build a recommendation request and explain the selected mode."""
    selected_mode = choose_recommendation_mode(
        user_rating_count=user_rating_count,
        requested_mode=requested_mode,
    )
    request = build_recommendation_request(
        user_id=user_id,
        mode=selected_mode,
        n=n,
    )

    normalized_mode = requested_mode.strip().lower()
    if normalized_mode != "auto":
        reason = f"requested mode '{selected_mode}' was selected explicitly."
    elif user_rating_count <= 0:
        reason = "auto selected content because the user has no training ratings."
    else:
        reason = (
            "auto selected hybrid because the user has "
            f"{user_rating_count} training ratings."
        )

    return request, reason


def recommend_item_cf_for_user(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    n: int,
    min_rating: float,
    min_common_users: int,
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


def recommend_for_request(
    request: RecommendationRequest,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    candidate_n: int,
    item_cf_weight: float,
    content_weight: float,
    min_rating: float,
    min_common_users: int,
) -> pd.DataFrame:
    """Execute the selected recommendation mode."""
    candidate_n = max(candidate_n, request.n)

    if request.mode == "content":
        return recommend_content_based(
            ratings=ratings,
            movies=movies,
            user_id=request.user_id,
            n=request.n,
        )

    if request.mode == "item_cf":
        return recommend_item_cf_for_user(
            ratings=ratings,
            movies=movies,
            user_id=request.user_id,
            n=request.n,
            min_rating=min_rating,
            min_common_users=min_common_users,
        )

    item_cf_recommendations = recommend_item_cf_for_user(
        ratings=ratings,
        movies=movies,
        user_id=request.user_id,
        n=candidate_n,
        min_rating=min_rating,
        min_common_users=min_common_users,
    )
    content_recommendations = recommend_content_based(
        ratings=ratings,
        movies=movies,
        user_id=request.user_id,
        n=candidate_n,
    )

    return recommend_hybrid(
        recommendation_frames=[
            standardize_recommendations(
                item_cf_recommendations,
                source="item_cf",
            ),
            standardize_recommendations(
                content_recommendations,
                source="content",
            ),
        ],
        source_weights={
            "item_cf": item_cf_weight,
            "content": content_weight,
        },
        n=request.n,
    )


def main() -> None:
    """Run recommendation through the strategy interface."""
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
        "--mode",
        choices=["auto", *list_recommendation_modes()],
        default="auto",
        help="Recommendation mode to run",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of recommendations to print",
    )
    parser.add_argument(
        "--candidate-n",
        type=int,
        default=25,
        help="Number of candidates collected per source for hybrid mode",
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

    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")
    movies = pd.read_csv(args.processed_directory / "movies.csv")
    user_rating_count = count_user_ratings(
        ratings=ratings,
        user_id=args.user_id,
    )
    request, reason = build_strategy_request_from_context(
        user_id=args.user_id,
        requested_mode=args.mode,
        n=args.n,
        user_rating_count=user_rating_count,
    )

    print(f"Recommendation strategy request for user {request.user_id}")
    print(f"Requested mode: {args.mode}")
    print(f"Selected mode: {request.mode}")
    print(f"Reason: {reason}")
    print(f"Training ratings for user: {user_rating_count}")
    print()

    if user_rating_count == 0:
        print(
            "No recommendations found because the current recommenders require "
            "at least one training rating for this user."
        )
        return

    recommendations = recommend_for_request(
        request=request,
        ratings=ratings,
        movies=movies,
        candidate_n=args.candidate_n,
        item_cf_weight=args.item_cf_weight,
        content_weight=args.content_weight,
        min_rating=args.min_rating,
        min_common_users=args.min_common_users,
    )

    if recommendations.empty:
        print("No recommendations found with the current settings.")
        return

    score_column = "hybrid_score" if request.mode == "hybrid" else "score"
    print(
        recommendations.to_string(
            index=False,
            formatters={score_column: "{:.3f}".format},
        )
    )


if __name__ == "__main__":
    main()
