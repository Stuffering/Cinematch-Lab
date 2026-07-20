import pandas as pd
import pytest

from cinematch.anomaly import (
    build_user_anomaly_features,
    detect_user_anomalies,
)


@pytest.mark.skip(reason="Stage 09 learning step: build user anomaly features")
def test_build_user_anomaly_features_summarizes_unusual_behavior() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "item_id": [10, 20, 10, 20, 30],
            "rating": [5, 1, 4, 4, 5],
        }
    )

    features = build_user_anomaly_features(ratings)

    assert features.index.tolist() == [1, 2]
    assert features.loc[1, "rating_count"] == 2
    assert features.loc[1, "rating_std"] > 0


@pytest.mark.skip(reason="Stage 09 learning step: detect user anomalies")
def test_detect_user_anomalies_flags_expected_number_of_users() -> None:
    features = pd.DataFrame(
        {
            "rating_count": [20, 22, 24, 400],
            "mean_rating": [3.5, 3.6, 3.4, 1.2],
            "rating_std": [0.8, 0.7, 0.9, 2.0],
        },
        index=[1, 2, 3, 4],
    )

    anomalies = detect_user_anomalies(
        features,
        contamination=0.25,
        random_state=42,
    )

    assert anomalies["user_id"].tolist() == [1, 2, 3, 4]
    assert anomalies["is_anomaly"].sum() == 1
