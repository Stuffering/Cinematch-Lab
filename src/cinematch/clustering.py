"""User clustering utilities."""

from __future__ import annotations

import pandas as pd


def build_user_clustering_features(
    ratings: pd.DataFrame,
    users: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row of clustering features per user."""
    raise NotImplementedError("Stage 08: build user clustering features")


def assign_user_segments(
    features: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign users to unsupervised behavior segments."""
    raise NotImplementedError("Stage 08: assign user segments")
