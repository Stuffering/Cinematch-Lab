"""Neural rating prediction utilities."""

from __future__ import annotations

import pandas as pd


def build_interaction_mappings(
    ratings: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build contiguous user and item index mappings for neural models."""
    raise NotImplementedError("Stage 10: build neural interaction mappings")


def encode_interactions(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Encode user-item interactions into model-ready integer indices."""
    raise NotImplementedError("Stage 10: encode neural interactions")


def build_neural_rating_model(
    n_users: int,
    n_items: int,
    embedding_dim: int = 16,
):
    """Build a neural rating prediction model."""
    raise NotImplementedError("Stage 10: build neural rating model")
