import pandas as pd
import pytest

from cinematch.baselines import (
    BiasBaseline,
    GlobalMeanBaseline,
    ItemMeanBaseline,
    UserMeanBaseline,
)


def test_global_mean_baseline_predicts_training_mean() -> None:
    train = pd.DataFrame({"rating": [5, 3, 4]})
    rows = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})

    model = GlobalMeanBaseline().fit(train)
    predictions = model.predict(rows)

    assert predictions.tolist() == [4.0, 4.0]


def test_user_mean_baseline_falls_back_to_global_mean() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 2],
        }
    )
    rows = pd.DataFrame({"user_id": [1, 2, 999], "item_id": [10, 20, 30]})

    model = UserMeanBaseline().fit(train)
    predictions = model.predict(rows)

    assert predictions.tolist() == [4.0, 2.0, pytest.approx(10 / 3)]

def test_item_mean_baseline_falls_back_to_global_mean() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 10, 20],
            "rating": [5, 3, 2],
        }
    )
    rows = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [10, 20, 999]})

    model = ItemMeanBaseline().fit(train)
    predictions = model.predict(rows)

    assert predictions.tolist() == [4.0, 2.0, pytest.approx(10 / 3)]


def test_bias_baseline_combines_global_user_and_item_biases() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "rating": [5, 3],
        }
    )
    rows = pd.DataFrame({"user_id": [1, 2, 999], "item_id": [10, 20, 999]})

    model = BiasBaseline(regularization=0.0, n_epochs=1).fit(train)
    predictions = model.predict(rows)

    assert predictions.tolist() == [5.0, 3.0, 4.0]
