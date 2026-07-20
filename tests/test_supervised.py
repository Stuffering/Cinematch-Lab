import pandas as pd

from cinematch.supervised import (
    add_rating_history_features,
    build_preprocessing_feature_lists,
    build_rating_feature_table,
    split_features_and_target,
)


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

    assert "user_id" not in features.columns
    assert "item_id" not in features.columns
    assert "rating" not in features.columns
    assert target.tolist() == [5]


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


def test_add_rating_history_features_uses_training_statistics() -> None:
    feature_table = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [5, 3],
        }
    )
    train_ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 4],
        }
    )

    enriched = add_rating_history_features(feature_table, train_ratings)

    assert enriched.loc[0, "user_rating_count"] == 2
    assert enriched.loc[0, "user_mean_rating"] == 4.0
    assert enriched.loc[0, "item_rating_count"] == 2
    assert enriched.loc[0, "item_mean_rating"] == 4.5
