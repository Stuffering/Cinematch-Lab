"""Hybrid recommendation utilities."""

from __future__ import annotations

import pandas as pd


def standardize_recommendations(
    recommendations: pd.DataFrame,
    source: str,
    score_column: str = "score",
) -> pd.DataFrame:
    """Convert recommendations into a common hybrid schema."""
    standardized = recommendations[["movie_id", "title", score_column]].copy()
    standardized = standardized.rename(columns={score_column: "source_score"})
    standardized["source"] = source

    return standardized[
        [
            "movie_id",
            "title",
            "source",
            "source_score",
        ]
    ]


def blend_recommendations(
    recommendation_frames: list[pd.DataFrame],
    source_weights: dict[str, float] | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Blend recommendation outputs from multiple sources."""
    if not recommendation_frames:
        return pd.DataFrame(
            columns=[
                "movie_id",
                "title",
                "hybrid_score",
            ]
        )

    combined = pd.concat(recommendation_frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(
            columns=[
                "movie_id",
                "title",
                "hybrid_score",
            ]
        )

    if source_weights is None:
        source_weights = {}

    combined["source_weight"] = combined["source"].map(source_weights).fillna(1.0)
    combined["weighted_score"] = (
        combined["source_score"] * combined["source_weight"]
    )

    blended = (
        combined.groupby(["movie_id", "title"], as_index=False)["weighted_score"]
        .sum()
        .rename(columns={"weighted_score": "hybrid_score"})
    )

    return (
        blended.sort_values(
            ["hybrid_score", "movie_id"],
            ascending=[False, True],
        )
        .head(n)
        .reset_index(drop=True)
    )


def recommend_hybrid(
    recommendation_frames: list[pd.DataFrame],
    source_weights: dict[str, float] | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Build a hybrid recommendation list from prepared recommendation frames."""
    return blend_recommendations(
        recommendation_frames=recommendation_frames,
        source_weights=source_weights,
        n=n,
    )
