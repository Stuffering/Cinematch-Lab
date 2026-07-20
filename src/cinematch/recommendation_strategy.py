"""Recommendation strategy interface utilities."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_RECOMMENDATION_MODES = {
    "hybrid": "Blend multiple recommendation sources into one ranked list.",
    "item_cf": "Recommend movies similar to items the user liked.",
    "content": "Recommend movies from explicit movie metadata preferences.",
}


@dataclass(frozen=True)
class RecommendationRequest:
    """Structured request for one recommendation workflow."""

    user_id: int
    mode: str
    n: int


def list_recommendation_modes() -> list[str]:
    """Return supported recommendation modes in stable UI order."""
    return [
        "hybrid",
        "item_cf",
        "content",
    ]


def validate_recommendation_mode(mode: str) -> str:
    """Validate and normalize a recommendation mode name."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in SUPPORTED_RECOMMENDATION_MODES:
        supported_modes = ", ".join(list_recommendation_modes())
        raise ValueError(
            f"Unsupported recommendation mode: {mode}. "
            f"Supported modes: {supported_modes}."
        )

    return normalized_mode


def build_recommendation_request(
    user_id: int,
    mode: str = "hybrid",
    n: int = 10,
) -> RecommendationRequest:
    """Build a validated recommendation request."""
    if user_id <= 0:
        raise ValueError("user_id must be a positive integer.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    return RecommendationRequest(
        user_id=user_id,
        mode=validate_recommendation_mode(mode),
        n=n,
    )


def choose_recommendation_mode(
    user_rating_count: int,
    requested_mode: str = "auto",
) -> str:
    """Choose a recommendation mode from user context and request preference."""
    normalized_mode = requested_mode.strip().lower()

    if normalized_mode != "auto":
        return validate_recommendation_mode(normalized_mode)

    if user_rating_count <= 0:
        return "content"

    return "hybrid"
