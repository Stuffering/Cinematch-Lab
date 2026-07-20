"""Anomaly detection utilities."""

from __future__ import annotations

import pandas as pd


def build_user_anomaly_features(
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Build user-level features for anomaly detection."""
    raise NotImplementedError("Stage 09: build user anomaly features")


def detect_user_anomalies(
    features: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Detect unusual users from behavior features."""
    raise NotImplementedError("Stage 09: detect user anomalies")
