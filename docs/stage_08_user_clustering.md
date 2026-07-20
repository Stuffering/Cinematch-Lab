# Stage 08 - User Clustering

This stage adds unsupervised learning for user segmentation.

The model idea is:

```text
Use user behavior features to group similar users without rating labels.
```

Earlier stages predicted ratings or recommended movies for a known user-item
target. Stage 08 changes the question: instead of predicting `rating`, it asks
whether users naturally form behavioral groups.

## Goal

Build a repeatable clustering workflow that:

1. Summarizes each user's rating behavior.
2. Optionally joins user metadata such as age, gender, and occupation.
3. Preprocesses numeric and categorical features.
4. Fits an unsupervised clustering model.
5. Interprets each segment with human-readable statistics.
6. Compares clustering outputs with earlier recommender features.

## Learning Path

1. Build one row of features per user.
2. Understand why clustering has no target `y`.
3. Scale numeric features before distance-based clustering.
4. Fit a simple K-Means model.
5. Inspect cluster sizes and segment profiles.
6. Decide whether the discovered groups are useful or only mathematical.

## Concepts

### Supervised vs Unsupervised Learning

Stage 07 used supervised regression:

```text
features X -> rating y
```

Stage 08 uses unsupervised clustering:

```text
features X -> segment labels discovered by the model
```

There is no known correct segment label in the raw data. The model creates
groups based on similarity in feature space.

### Feature Scale

K-Means uses distances. A large-scale column such as `rating_count` can dominate
a smaller-scale column such as `mean_rating`, so scaling is part of the model
logic rather than cosmetic preprocessing.

### Interpretation

A cluster label such as `segment=2` is not meaningful by itself. The useful part
is the profile behind it, for example:

```text
frequent raters, high average ratings, action-heavy preferences
```

## Acceptance Checks

Stage 08 starts with a learning scaffold. Unskip and complete tests as each
piece is implemented:

```bash
python -m pytest tests/test_clustering.py -v
python -m pytest -q
python -m ruff check .
```
