"""Cluster users into behavior segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.clustering import (
    assign_user_segments,
    build_user_clustering_features,
    score_user_segments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Cluster users and print segment assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory containing prepared MovieLens CSV files",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of user segments to discover; Stage 08 default is 4",
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")
    users = pd.read_csv(args.processed_directory / "users.csv")

    features = build_user_clustering_features(ratings, users)
    numeric_features = features.select_dtypes(include="number")
    print(
        "Features used for clustering:",
        ", ".join(numeric_features.columns),
    )
    print()

    segments = assign_user_segments(
        numeric_features,
        n_clusters=args.n_clusters,
    )
    silhouette = score_user_segments(numeric_features, segments)

    profile_table = segments.join(features, on="user_id")

    segment_profiles = (
        profile_table.groupby("segment")
        .agg(
            user_count=("user_id", "size"),
            avg_rating_count=("rating_count", "mean"),
            avg_mean_rating=("mean_rating", "mean"),
            avg_age=("age", "mean"),
        )
        .reset_index()
        .sort_values("segment")
    )

    top_occupations = (
        profile_table.groupby("segment")["occupation"]
        .agg(lambda values: values.value_counts().idxmax())
        .rename("top_occupation")
        .reset_index()
    )

    segment_profiles = segment_profiles.merge(
        top_occupations,
        on="segment",
        how="left",
    )

    print(f"Silhouette score: {silhouette:.3f}")
    print()

    print("Segment profiles:")
    print(segment_profiles.round(3).to_string(index=False))

    print("\nSample user assignments:")
    print(segments.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
