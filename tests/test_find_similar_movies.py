import pandas as pd
import pytest

from scripts.find_similar_movies import find_similar_movies


def test_find_similar_movies_returns_titles_and_overlap_counts() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 20, 30, 10, 20, 30, 10, 20, 30],
            "rating": [5, 5, 1, 4, 4, 2, 1, 1, 5],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "title": ["Query Movie", "Similar Movie", "Opposite Movie"],
        }
    )

    similar = find_similar_movies(
        ratings=ratings,
        movies=movies,
        item_id=10,
        n=1,
        min_common_users=3,
    )

    assert similar["movie_id"].tolist() == [20]
    assert similar["title"].tolist() == ["Similar Movie"]
    assert similar["similarity"].tolist() == [pytest.approx(1.0)]
    assert similar["common_users"].tolist() == [3]


def test_find_similar_movies_rejects_unknown_item() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5],
        }
    )
    movies = pd.DataFrame({"movie_id": [10], "title": ["Known Movie"]})

    with pytest.raises(ValueError, match="item_id 999"):
        find_similar_movies(
            ratings=ratings,
            movies=movies,
            item_id=999,
        )
