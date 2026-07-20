import pandas as pd

from cinematch.hybrid import (
    blend_recommendations,
    recommend_hybrid,
    standardize_recommendations,
)


def test_standardize_recommendations_adds_source_and_renames_score() -> None:
    recommendations = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Movie A", "Movie B"],
            "similarity": [0.9, 0.5],
        }
    )

    standardized = standardize_recommendations(
        recommendations,
        source="item_cf",
        score_column="similarity",
    )

    assert standardized.columns.tolist() == [
        "movie_id",
        "title",
        "source",
        "source_score",
    ]
    assert standardized["source"].tolist() == ["item_cf", "item_cf"]
    assert standardized["source_score"].tolist() == [0.9, 0.5]


def test_blend_recommendations_combines_weighted_sources() -> None:
    item_cf = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["Movie A", "Movie B"],
            "source": ["item_cf", "item_cf"],
            "source_score": [1.0, 0.5],
        }
    )
    content = pd.DataFrame(
        {
            "movie_id": [10, 30],
            "title": ["Movie A", "Movie C"],
            "source": ["content", "content"],
            "source_score": [0.5, 1.0],
        }
    )

    blended = blend_recommendations(
        [item_cf, content],
        source_weights={"item_cf": 0.7, "content": 0.3},
        n=3,
    )

    assert blended["movie_id"].tolist() == [10, 20, 30]
    assert blended.loc[0, "hybrid_score"] == 0.85


def test_recommend_hybrid_returns_top_n_blended_movies() -> None:
    recommendations = [
        pd.DataFrame(
            {
                "movie_id": [10, 20],
                "title": ["Movie A", "Movie B"],
                "source": ["content", "content"],
                "source_score": [0.3, 1.0],
            }
        )
    ]

    hybrid = recommend_hybrid(recommendations, n=1)

    assert hybrid["movie_id"].tolist() == [20]
