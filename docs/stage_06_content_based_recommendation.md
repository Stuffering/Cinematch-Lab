# Stage 06 - Content-Based Recommendation

This stage adds content-based recommendation to CineMatch Lab.

The model idea is:

```text
Recommend movies whose content features match a user's historical preferences.
```

Stage 04 used item-item collaborative similarity from rating behavior. Stage 05
learned latent user and movie factors from rating errors. Stage 06 returns to
explicit movie metadata, starting with MovieLens genre columns.

## Goal

Build a content-based recommender that:

1. Extracts movie genre feature vectors.
2. Builds a user preference profile from movies the user has rated.
3. Scores unseen movies by matching movie features to the user profile.
4. Compares content-based recommendations with collaborative approaches.

## Learning Path

1. Identify genre feature columns in the movies table.
2. Build a movie-feature matrix indexed by `movie_id`.
3. Convert one user's rating history into a weighted genre preference profile.
4. Score candidate movies with a dot product or cosine similarity.
5. Exclude movies the user has already rated.
6. Return ranked recommendations with movie titles.
7. Compare the strengths and weaknesses of content-based and collaborative
   filtering.

## Concepts

### Explicit Features

Unlike latent factors, genre columns are visible metadata:

```text
Action, Comedy, Drama, Sci-Fi, ...
```

The model does not need other users to rate the same movies to make a
recommendation. It can recommend from movie attributes directly.

### User Profile

A user profile summarizes what kinds of movie features a user tends to like.
For example, if a user gives high ratings to `Sci-Fi` and `Action` movies, the
profile should assign higher weight to those genres.

### Content Score

The simplest score is a dot product:

```text
score(user, movie) = dot(user_profile, movie_features)
```

Higher scores mean the movie's explicit content features better match the
user's learned profile.

## Acceptance Checks

Run the content-based recommendation tests:

```bash
python -m pytest tests/test_content.py -v
python -m pytest tests/test_recommend_content.py -v
python -m pytest -q
python -m ruff check .
```

To run content-based recommendations on the prepared MovieLens data:

```bash
python scripts/recommend_content.py --user-id 1 --n 10
```

The script also prints the strongest positive and negative genre preferences
from the centered user profile so recommendations are easier to explain.

The first target is not to beat matrix factorization. The first target is to
make feature-based recommendations explainable:

```text
This movie is recommended because the user tends to like these genres.
```
