"""Anomaly detection utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def build_user_anomaly_features(
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Build user-level features for anomaly detection."""
    features = ratings.groupby("user_id")["rating"].agg(
        rating_count="size",
        mean_rating="mean",
        rating_std="std",
        min_rating="min",
        max_rating="max",
    )

    features["rating_std"] = features["rating_std"].fillna(0.0)

    return features.sort_index()


def detect_user_anomalies(
    features: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Detect unusual users from behavior features."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )
    labels = model.fit_predict(scaled_features)
    scores = -model.score_samples(scaled_features)

    return pd.DataFrame(
        {
            "user_id": features.index,
            "anomaly_score": scores,
            "is_anomaly": labels == -1,
        }
    )
