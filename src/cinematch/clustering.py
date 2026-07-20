"""User clustering utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_user_clustering_features(
    ratings: pd.DataFrame,
    users: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row of clustering features per user."""
    features = ratings.groupby("user_id")["rating"].agg(
        rating_count="size",
        mean_rating="mean",
    )

    if users is not None:
        user_metadata = users.set_index("user_id")
        features = features.join(user_metadata, how="left")

    return features.sort_index()


def assign_user_segments(
    features: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign users to unsupervised behavior segments."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = model.fit_predict(scaled_features)

    return pd.DataFrame(
        {
            "user_id": features.index,
            "segment": labels,
        }
    )


def score_user_segments(
    features: pd.DataFrame,
    segments: pd.DataFrame,
) -> float:
    """Score user segments with silhouette score."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return float(silhouette_score(scaled_features, segments["segment"]))
