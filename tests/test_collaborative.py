import pandas as pd
import pytest

from cinematch.collaborative import (
    build_user_item_matrix,
    center_user_ratings,
    compute_item_similarity_matrix,
    cosine_similarity,
    predict_rating_for_user_item,
    predict_ratings_item_based,
    recommend_items_for_user,
    top_similar_items,
)


def test_build_user_item_matrix_creates_user_rows_and_item_columns() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 4],
        }
    )

    matrix = build_user_item_matrix(ratings)

    assert matrix.index.tolist() == [1, 2]
    assert matrix.columns.tolist() == [10, 20]
    assert matrix.loc[1, 10] == 5
    assert matrix.loc[1, 20] == 3
    assert matrix.loc[2, 10] == 4
    assert pd.isna(matrix.loc[2, 20])


def test_center_user_ratings_subtracts_each_users_observed_mean() -> None:
    matrix = pd.DataFrame(
        {
            10: [5.0, 4.0],
            20: [3.0, None],
        },
        index=[1, 2],
    )

    centered = center_user_ratings(matrix)

    assert centered.loc[1, 10] == pytest.approx(1.0)
    assert centered.loc[1, 20] == pytest.approx(-1.0)
    assert centered.loc[2, 10] == pytest.approx(0.0)
    assert pd.isna(centered.loc[2, 20])


def test_cosine_similarity_compares_two_vectors() -> None:
    vector_a = pd.Series([1.0, 0.0, -1.0])
    vector_b = pd.Series([1.0, 0.0, -1.0])
    vector_c = pd.Series([-1.0, 0.0, 1.0])

    assert cosine_similarity(vector_a, vector_b) == pytest.approx(1.0)
    assert cosine_similarity(vector_a, vector_c) == pytest.approx(-1.0)


def test_compute_item_similarity_matrix_returns_square_item_matrix() -> None:
    centered = pd.DataFrame(
        {
            10: [1.0, 0.0, -1.0],
            20: [1.0, 0.0, -1.0],
            30: [-1.0, 0.0, 1.0],
        },
        index=[1, 2, 3],
    )

    similarity = compute_item_similarity_matrix(centered)

    assert similarity.index.tolist() == [10, 20, 30]
    assert similarity.columns.tolist() == [10, 20, 30]
    assert similarity.loc[10, 10] == pytest.approx(1.0)
    assert similarity.loc[10, 20] == pytest.approx(1.0)
    assert similarity.loc[10, 30] == pytest.approx(-1.0)


def test_top_similar_items_excludes_query_item_and_sorts_descending() -> None:
    similarity = pd.DataFrame(
        {
            10: [1.0, 0.8, 0.2],
            20: [0.8, 1.0, 0.5],
            30: [0.2, 0.5, 1.0],
        },
        index=[10, 20, 30],
    )

    similar = top_similar_items(similarity, item_id=10, n=2)

    assert similar.index.tolist() == [20, 30]
    assert similar.tolist() == [0.8, 0.2]


def test_recommend_items_for_user_scores_unseen_similar_items() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 10, 20, 30, 10, 20, 30],
            "rating": [5, 5, 5, 1, 1, 1, 5],
        }
    )
    user_item_matrix = build_user_item_matrix(ratings)
    centered_matrix = center_user_ratings(user_item_matrix)

    recommendations = recommend_items_for_user(
        ratings=ratings,
        centered_matrix=centered_matrix,
        user_id=1,
        n=2,
        min_rating=4,
        min_common_users=2,
    )

    assert recommendations.index.tolist() == [20]
    assert recommendations.loc[20] == pytest.approx(5.0)


def test_recommend_items_for_user_rejects_unknown_user() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
        }
    )
    centered_matrix = pd.DataFrame({10: [0.0]}, index=[1])

    with pytest.raises(ValueError, match="user_id 999"):
        recommend_items_for_user(
            ratings=ratings,
            centered_matrix=centered_matrix,
            user_id=999,
        )


def test_predict_rating_for_user_item_uses_similar_centered_ratings() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 20, 10, 20, 30, 10, 20, 30],
            "rating": [5, 3, 5, 3, 5, 1, 5, 1],
        }
    )
    user_item_matrix = build_user_item_matrix(ratings)
    centered_matrix = center_user_ratings(user_item_matrix)

    prediction = predict_rating_for_user_item(
        ratings=ratings,
        centered_matrix=centered_matrix,
        user_id=1,
        item_id=30,
        min_common_users=2,
    )

    assert prediction == pytest.approx(5.0)


def test_predict_rating_for_user_item_falls_back_to_user_mean() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "item_id": [10, 20, 10, 20, 30],
            "rating": [5, 3, 5, 3, 5],
        }
    )
    user_item_matrix = build_user_item_matrix(ratings)
    centered_matrix = center_user_ratings(user_item_matrix)

    prediction = predict_rating_for_user_item(
        ratings=ratings,
        centered_matrix=centered_matrix,
        user_id=1,
        item_id=30,
        min_common_users=99,
    )

    assert prediction == pytest.approx(4.0)


def test_predict_ratings_item_based_returns_aligned_series() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 20, 10, 20, 30, 10, 20, 30],
            "rating": [5, 3, 5, 3, 5, 1, 5, 1],
        }
    )
    rows_to_predict = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [30, 30],
        },
        index=[10, 20],
    )

    predictions = predict_ratings_item_based(
        train_ratings=train,
        rows_to_predict=rows_to_predict,
        min_common_users=2,
    )

    assert predictions.name == "prediction"
    assert predictions.index.tolist() == [10, 20]
    assert predictions.tolist() == [pytest.approx(5.0), pytest.approx(5.0)]


def test_predict_ratings_item_based_falls_back_for_cold_start_rows() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [5, 3, 4],
        }
    )
    rows_to_predict = pd.DataFrame(
        {
            "user_id": [1, 999],
            "item_id": [999, 10],
        }
    )

    predictions = predict_ratings_item_based(
        train_ratings=train,
        rows_to_predict=rows_to_predict,
    )

    assert predictions.tolist() == [pytest.approx(4.0), pytest.approx(4.0)]
