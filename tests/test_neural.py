import pandas as pd

from cinematch.neural import (
    build_interaction_mappings,
    build_neural_rating_model,
    encode_interactions,
)


def test_build_interaction_mappings_are_contiguous() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [10, 10, 20],
            "item_id": [100, 200, 100],
            "rating": [5, 3, 4],
        }
    )

    user_to_index, item_to_index = build_interaction_mappings(ratings)

    assert user_to_index == {10: 0, 20: 1}
    assert item_to_index == {100: 0, 200: 1}


def test_encode_interactions_returns_features_and_target() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [10, 20],
            "item_id": [100, 200],
            "rating": [5, 3],
        }
    )

    features, target = encode_interactions(
        ratings,
        user_to_index={10: 0, 20: 1},
        item_to_index={100: 0, 200: 1},
    )

    assert features.to_dict(orient="list") == {
        "user_index": [0, 1],
        "item_index": [0, 1],
    }
    assert target.tolist() == [5, 3]


def test_build_neural_rating_model_compiles() -> None:
    model = build_neural_rating_model(
        n_users=2,
        n_items=3,
        embedding_dim=4,
    )

    assert model.optimizer is not None
    assert model.loss == "mse"
