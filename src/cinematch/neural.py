"""Neural rating prediction utilities."""

from __future__ import annotations

import pandas as pd
import tensorflow as tf


def build_interaction_mappings(
    ratings: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build contiguous user and item index mappings for neural models."""
    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }
    item_to_index = {
        item_id: index
        for index, item_id in enumerate(item_ids)
    }

    return user_to_index, item_to_index


def encode_interactions(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Encode user-item interactions into model-ready integer indices."""
    features = pd.DataFrame(
        {
            "user_index": ratings["user_id"].map(user_to_index),
            "item_index": ratings["item_id"].map(item_to_index),
        }
    )
    target = ratings["rating"]

    return features, target


def build_neural_rating_model(
    n_users: int,
    n_items: int,
    embedding_dim: int = 16,
) -> tf.keras.Model:
    """Build a neural rating prediction model."""
    user_input = tf.keras.Input(shape=(1,), name="user_index")
    item_input = tf.keras.Input(shape=(1,), name="item_index")

    user_embedding = tf.keras.layers.Embedding(
        input_dim=n_users,
        output_dim=embedding_dim,
        name="user_embedding",
    )(user_input)
    item_embedding = tf.keras.layers.Embedding(
        input_dim=n_items,
        output_dim=embedding_dim,
        name="item_embedding",
    )(item_input)

    user_vector = tf.keras.layers.Flatten(name="user_vector")(user_embedding)
    item_vector = tf.keras.layers.Flatten(name="item_vector")(item_embedding)

    prediction = tf.keras.layers.Dot(
        axes=1,
        name="rating_prediction",
    )([user_vector, item_vector])

    model = tf.keras.Model(
        inputs={
            "user_index": user_input,
            "item_index": item_input,
        },
        outputs=prediction,
    )
    model.compile(
        optimizer="adam",
        loss="mse",
    )

    return model
