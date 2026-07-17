import pandas as pd

from scripts.recommend_movies import recommend_movies_for_user


def test_recommend_movies_for_user_adds_movie_titles() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 2, 3, 3, 3],
            "item_id": [10, 10, 20, 30, 10, 20, 30],
            "rating": [5, 5, 5, 1, 1, 1, 5],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "title": ["Watched Movie", "Recommended Movie", "Opposite Movie"],
        }
    )

    recommendations = recommend_movies_for_user(
        ratings=ratings,
        movies=movies,
        user_id=1,
        n=1,
        min_rating=4,
        min_common_users=2,
    )

    assert recommendations["movie_id"].tolist() == [20]
    assert recommendations["title"].tolist() == ["Recommended Movie"]
