import pandas as pd
import pytest

from cinematch.clustering import (
    assign_user_segments,
    build_user_clustering_features,
)


@pytest.mark.skip(reason="Stage 08 learning step: build user clustering features")
def test_build_user_clustering_features_summarizes_rating_behavior() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 4],
        }
    )

    features = build_user_clustering_features(ratings)

    assert features.index.tolist() == [1, 2]
    assert features.loc[1, "rating_count"] == 2
    assert features.loc[1, "mean_rating"] == 4.0


@pytest.mark.skip(reason="Stage 08 learning step: join user metadata")
def test_build_user_clustering_features_can_join_user_metadata() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [5, 3],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [1, 2],
            "age": [25, 40],
            "gender": ["M", "F"],
            "occupation": ["student", "engineer"],
        }
    )

    features = build_user_clustering_features(ratings, users)

    assert features.loc[1, "age"] == 25
    assert features.loc[2, "occupation"] == "engineer"


@pytest.mark.skip(reason="Stage 08 learning step: fit user segments")
def test_assign_user_segments_returns_one_label_per_user() -> None:
    features = pd.DataFrame(
        {
            "rating_count": [10, 12, 80, 90],
            "mean_rating": [3.0, 3.2, 4.5, 4.7],
        },
        index=[1, 2, 3, 4],
    )

    segments = assign_user_segments(features, n_clusters=2, random_state=42)

    assert segments["user_id"].tolist() == [1, 2, 3, 4]
    assert segments["segment"].nunique() == 2
