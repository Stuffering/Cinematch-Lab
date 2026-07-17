import pandas as pd
import pytest

from cinematch.matrix_factorization import MatrixFactorizationModel


def test_matrix_factorization_model_stores_configuration() -> None:
    model = MatrixFactorizationModel(
        n_factors=3,
        learning_rate=0.02,
        regularization=0.5,
        n_epochs=7,
        random_state=123,
    )

    assert model.n_factors == 3
    assert model.learning_rate == 0.02
    assert model.regularization == 0.5
    assert model.n_epochs == 7
    assert model.random_state == 123


@pytest.mark.skip(reason="Stage 05 TODO(student): implement fit mappings")
def test_fit_learns_user_and_item_mappings() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 4],
        }
    )

    model = MatrixFactorizationModel(n_factors=2).fit(ratings)

    assert model.global_mean_ == pytest.approx(4.0)
    assert model.user_index_ == {1: 0, 2: 1}
    assert model.item_index_ == {10: 0, 20: 1}


@pytest.mark.skip(reason="Stage 05 TODO(student): implement prediction")
def test_predict_returns_aligned_series() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 4],
        }
    )
    rows_to_predict = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
        },
        index=[10, 20],
    )

    model = MatrixFactorizationModel(n_factors=2, n_epochs=2).fit(ratings)
    predictions = model.predict(rows_to_predict)

    assert predictions.name == "prediction"
    assert predictions.index.tolist() == [10, 20]
    assert len(predictions) == 2


@pytest.mark.skip(reason="Stage 05 TODO(student): implement SGD training")
def test_matrix_factorization_can_overfit_tiny_pattern() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 10, 20],
            "rating": [5, 1, 1, 5],
        }
    )

    model = MatrixFactorizationModel(
        n_factors=2,
        learning_rate=0.05,
        regularization=0.01,
        n_epochs=200,
        random_state=42,
    ).fit(ratings)
    predictions = model.predict(ratings)

    assert predictions.tolist() == pytest.approx(ratings["rating"].tolist(), abs=0.5)
