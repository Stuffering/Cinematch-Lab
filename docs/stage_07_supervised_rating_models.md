# Stage 07 - Supervised Rating Models

This stage adds supervised learning models for rating prediction.

The model idea is:

```text
Use explicit user, movie, and interaction features to predict rating.
```

Stage 05 learned latent factors from user-item interactions. Stage 06 used
explicit movie content features for explainable recommendations. Stage 07 turns
rating prediction into a supervised regression problem using scikit-learn.

## Goal

Build a repeatable supervised learning workflow that:

1. Joins ratings with user metadata and movie metadata.
2. Converts raw tables into model-ready features.
3. Trains a regression model on `ratings_train.csv`.
4. Tunes model choices on `ratings_valid.csv`.
5. Checks the final selected model on `ratings_test.csv`.
6. Compares results against the Stage 03, Stage 04, and Stage 05 models.

## Learning Path

1. Build a rating feature table from ratings, users, and movies.
2. Separate feature columns from the target `rating`.
3. Encode categorical columns such as gender and occupation.
4. Train a simple supervised regression model.
5. Evaluate with RMSE and MAE.
6. Compare supervised features with latent factor methods.
7. Decide whether explicit metadata improves rating prediction.

## Concepts

### Supervised Regression

The target is the observed rating:

```text
y = rating
```

The input features can include:

```text
user age, gender, occupation
movie genres
rating timestamp features
user/movie historical aggregates
```

### Explicit Features vs Latent Factors

Stage 05 learned hidden user and movie vectors. Stage 07 uses visible features
that humans can inspect directly. This makes the model easier to diagnose, but
it may miss hidden taste patterns captured by matrix factorization.

### Model Comparison

The current strongest rating predictor is Stage 05 matrix factorization:

```text
valid RMSE: 1.082666
valid MAE:  0.857215

test RMSE:  1.030065
test MAE:   0.828554
```

Stage 07 should first aim to establish a clear supervised learning baseline,
then compare whether explicit metadata can improve on the learned latent-factor
model.

## Acceptance Checks

Stage 07 starts with a learning scaffold. Unskip and complete tests as each
piece is implemented:

```bash
python -m pytest tests/test_supervised.py -v
python -m pytest -q
python -m ruff check .
```
