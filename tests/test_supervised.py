import pandas as pd
import pytest

from cinematch.supervised import (
    build_preprocessing_feature_lists,
    build_rating_feature_table,
    split_features_and_target,
)


@pytest.mark.skip(reason="Stage 07 learning step: build feature table")
def test_build_rating_feature_table_joins_user_and_movie_metadata() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
            "timestamp": [1_000],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [1],
            "age": [25],
            "gender": ["M"],
            "occupation": ["student"],
            "zip_code": ["12345"],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10],
            "title": ["Example Movie"],
            "Action": [1],
            "Comedy": [0],
        }
    )

    feature_table = build_rating_feature_table(ratings, users, movies)

    assert feature_table.loc[0, "rating"] == 5
    assert feature_table.loc[0, "age"] == 25
    assert feature_table.loc[0, "title"] == "Example Movie"
    assert feature_table.loc[0, "Action"] == 1


@pytest.mark.skip(reason="Stage 07 learning step: split features and target")
def test_split_features_and_target_removes_rating_column() -> None:
    feature_table = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
            "age": [25],
        }
    )

    features, target = split_features_and_target(feature_table)

    assert "rating" not in features.columns
    assert target.tolist() == [5]


@pytest.mark.skip(reason="Stage 07 learning step: identify feature types")
def test_build_preprocessing_feature_lists_separates_numeric_and_categorical() -> None:
    features = pd.DataFrame(
        {
            "age": [25],
            "gender": ["M"],
            "occupation": ["student"],
            "Action": [1],
        }
    )

    numeric_features, categorical_features = build_preprocessing_feature_lists(
        features
    )

    assert "age" in numeric_features
    assert "Action" in numeric_features
    assert "gender" in categorical_features
    assert "occupation" in categorical_features
