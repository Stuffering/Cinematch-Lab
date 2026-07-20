"""Hybrid recommendation utilities."""

from __future__ import annotations

import pandas as pd


def standardize_recommendations(
    recommendations: pd.DataFrame,
    source: str,
    score_column: str = "score",
) -> pd.DataFrame:
    """Convert recommendations into a common hybrid schema."""
    raise NotImplementedError("Stage 11: standardize recommendation outputs")


def blend_recommendations(
    recommendation_frames: list[pd.DataFrame],
    source_weights: dict[str, float] | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Blend recommendation outputs from multiple sources."""
    raise NotImplementedError("Stage 11: blend recommendation outputs")


def recommend_hybrid(
    recommendation_frames: list[pd.DataFrame],
    source_weights: dict[str, float] | None = None,
    n: int = 10,
) -> pd.DataFrame:
    """Build a hybrid recommendation list from prepared recommendation frames."""
    raise NotImplementedError("Stage 11: build hybrid recommendations")
