import pandas as pd

from scripts.recommend_strategy import (
    build_strategy_request_from_context,
    count_user_ratings,
)


def test_count_user_ratings_returns_training_history_size() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 30],
            "rating": [5, 4, 3],
        }
    )

    assert count_user_ratings(ratings, user_id=1) == 2
    assert count_user_ratings(ratings, user_id=99) == 0


def test_build_strategy_request_from_context_explains_auto_mode() -> None:
    request, reason = build_strategy_request_from_context(
        user_id=1,
        requested_mode="auto",
        n=5,
        user_rating_count=20,
    )

    assert request.user_id == 1
    assert request.mode == "hybrid"
    assert request.n == 5
    assert "auto selected hybrid" in reason
