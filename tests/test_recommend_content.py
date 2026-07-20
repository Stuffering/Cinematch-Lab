import pandas as pd

from scripts.recommend_content import (
    recommend_content_for_user,
    summarize_user_content_profile,
)


def test_recommend_content_for_user_returns_titles_and_scores() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "rating": [5, 1],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "title": ["Seen Action", "Seen Comedy", "Unseen Action"],
            "Action": [1, 0, 1],
            "Comedy": [0, 1, 0],
        }
    )

    recommendations = recommend_content_for_user(
        ratings=ratings,
        movies=movies,
        user_id=1,
        n=1,
    )

    assert recommendations["movie_id"].tolist() == [30]
    assert recommendations["title"].tolist() == ["Unseen Action"]
    assert recommendations["score"].iloc[0] > 0


def test_summarize_user_content_profile_splits_positive_and_negative_genres() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "rating": [5, 1],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Seen Action", "Seen Comedy"],
            "Action": [1, 0],
            "Comedy": [0, 1],
        }
    )

    positive, negative = summarize_user_content_profile(
        ratings=ratings,
        movies=movies,
        user_id=1,
        n=1,
    )

    assert positive.index.tolist() == ["Action"]
    assert negative.index.tolist() == ["Comedy"]
