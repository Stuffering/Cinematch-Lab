# Stage 04 - Item-Based Collaborative Filtering

This stage moves CineMatch Lab beyond simple rating baselines and introduces
the first true recommendation algorithm: item-item collaborative filtering.

The model idea is:

```text
Movies are similar if users rate them in similar patterns.
```

Instead of only asking whether a movie is generally popular, this stage asks:

- Which movies receive similar rating patterns from the same users?
- How do we convert long-form ratings into a user-item matrix?
- Why do we center user ratings before comparing movies?
- How is cosine similarity computed?
- How can similar movies become recommendations?

## Scope

Stage 04 starts with item similarity, then extends it into item-based rating
prediction and user-level recommendation.

Student-owned implementation lives in:

```text
src/cinematch/collaborative.py
tests/test_collaborative.py
```

## Learning Path

1. Build a user-item matrix from `ratings_train.csv`.
2. Mean-center each user's observed ratings.
3. Compute cosine similarity between two movie columns.
4. Expand pairwise cosine similarity into an item-item matrix.
5. Retrieve the top similar movies for a selected movie.
6. Run similar-movie lookup on the real MovieLens training split.
7. Recommend unseen movies for a user from movies they rated highly.
8. Predict a user's rating for one target movie from similar rated movies.
9. Later, evaluate recommendation quality and compare against rating baselines.

## Concepts

### User-Item Matrix

Raw ratings are stored in long format:

```text
user_id | item_id | rating
```

Collaborative filtering usually works with a matrix:

```text
         movie 10   movie 20   movie 30
user 1      5          3          NaN
user 2      4         NaN          2
```

Missing values mean "not rated", not zero.

### User-Centered Ratings

Users have different scoring habits. Some users rate generously; others rate
strictly. User-centering subtracts each user's average observed rating so that
movie similarity focuses on preference patterns instead of raw score level.

Example:

```text
raw ratings:      [5, 3]
user mean:         4
centered ratings: [1, -1]
```

### Cosine Similarity

Cosine similarity compares vector direction:

```text
similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
```

For movie columns, it measures whether users tend to rate two movies above or
below their personal average in similar ways.

### Rating Prediction

The item-based prediction formula uses the user's own mean rating as the
baseline, then adds a similarity-weighted centered preference term:

```text
pred(u, i)
= mean_rating(u)
+ sum(sim(i, k) * (rating(u, k) - mean_rating(u)))
  / sum(abs(sim(i, k)))
```

Where:

- `u` is the target user.
- `i` is the target movie.
- `k` is a movie already rated by user `u`.
- `rating(u, k) - mean_rating(u)` is the user's centered rating for movie `k`.
- `sim(i, k)` is the cosine similarity between the target movie and a movie the
  user already rated.

If no usable similar movies are found, the prediction falls back to the user's
mean rating. Batch evaluation precomputes item-pair similarity and common-user
overlap matrices, so validation rows can reuse fast matrix lookups instead of
recomputing cosine similarity for every prediction.

## Acceptance Checks

As each TODO is implemented, unskip the matching test. To run the real-data
similar movie lookup:

```bash
python scripts/find_similar_movies.py --item-id 50 --n 10
```

Movie ID `50` is `Star Wars (1977)` in MovieLens 100K. The script defaults to
requiring at least 20 users who rated both movies to reduce small-sample noise.

To generate user-level recommendations:

```bash
python scripts/recommend_movies.py --user-id 1 --n 10
```

The current recommendation score is a similarity-weighted sum of the user's
ratings for movies they liked. It is a simple first ranking score, not yet a
fully evaluated rating prediction model.

To evaluate rating prediction:

```bash
python scripts/evaluate_item.py --eval-split valid
python scripts/evaluate_item.py --eval-split test
```

Use `--max-rows` for faster learning checks while editing:

```bash
python scripts/evaluate_item.py --eval-split valid --max-rows 2000
```

Current full-split results:

```text
 split  rows     rmse      mae
 valid 20000 1.177426 0.966152
  test 20000 1.110922 0.931586
```

Parameter tuning checked `min_common_users` values of `1, 3, 5, 10, 20, 50`
and `min_similarity` values of `0.00, 0.05, 0.10, 0.20, 0.30` on the full
validation split.

Best validation result:

```text
 min_common_users  min_similarity     rmse      mae fallback_rate
                1            0.05 1.175697 0.966276        83.17%
```

Default test confirmation:

```text
 min_common_users  min_similarity     rmse      mae
                5            0.00 1.110922 0.931586
```

Best-validation test confirmation:

```text
 min_common_users  min_similarity    rmse      mae
                1            0.05 1.11331 0.934346
```

This item-based CF model is now evaluable, but it does not beat the Stage 03
regularized bias baseline yet. The default parameters are kept because the
validation improvement from more permissive filtering did not hold on the test
split.

Run these checks before committing:

```bash
python -m pytest tests/test_collaborative.py -v
python -m pytest tests/test_find_similar_movies.py -v
python -m pytest tests/test_recommend_movies.py -v
python scripts/evaluate_item.py --eval-split valid
python -m pytest -q
python -m ruff check .
```

Keep the implementation readable before adding more complex model tuning.
