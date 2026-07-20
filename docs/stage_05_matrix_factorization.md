# Stage 05 - Matrix Factorization

This stage moves CineMatch Lab from similarity lookup to a learned latent
factor recommendation model.

The model idea is:

```text
Users and movies can both be represented by learned vectors.
The predicted rating comes from how well those vectors match.
```

Stage 04 compared movies directly through cosine similarity over centered
rating vectors. Stage 05 learns hidden user and item factors from rating errors.

## Goal

Build a small, readable matrix factorization model that predicts ratings with:

```text
prediction(u, i)
= global_mean
+ user_bias(u)
+ item_bias(i)
+ dot(user_factors(u), item_factors(i))
```

The model should be trained only on `ratings_train.csv`, tuned on
`ratings_valid.csv`, and checked on `ratings_test.csv` after the validation
choice is made.

## Learning Path

1. Create user and item ID mappings.
2. Initialize user and item latent factor matrices.
3. Start from the Stage 03 bias baseline formula.
4. Add the latent factor dot product.
5. Define the squared-error loss with regularization.
6. Implement one stochastic gradient descent update.
7. Repeat updates across multiple epochs.
8. Predict ratings for validation rows.
9. Compare against the Stage 03 bias baseline and Stage 04 item-CF results.

## Concepts

### Latent Factors

A latent factor is a learned hidden dimension. It is not a raw column from the
MovieLens data. For example, a factor might implicitly represent how much a user
likes action-heavy movies, older movies, romantic movies, or some mixture the
model discovers from rating behavior.

### Dot Product

The dot product measures the match between a user vector and a movie vector:

```text
dot(user_factors, item_factors)
```

If the user and movie point in similar learned directions, the dot product
increases the predicted rating. If they point in opposite directions, it lowers
the predicted rating.

### Loss Function

The training objective is to make prediction errors small while keeping
parameters controlled:

```text
sum((rating - prediction) ** 2)
+ regularization * parameter_size_penalty
```

This connects directly to Stage 03 regularized bias learning: regularization
does not get added to the final prediction. It is used during training to keep
learned parameters from becoming too large.

### SGD Update

Stochastic gradient descent updates parameters one observed rating at a time.
For each row:

```text
error = actual_rating - predicted_rating
```

Then the model nudges user bias, item bias, user factors, and item factors in
the direction that reduces that error.

## Acceptance Checks

Run the unit tests and full project checks:

```bash
python -m pytest tests/test_matrix_factorization.py -v
python -m pytest -q
python -m ruff check .
```

Train and evaluate the selected matrix factorization configuration:

```bash
python scripts/train_matrix_factorization.py --eval-split valid
python scripts/train_matrix_factorization.py --eval-split test
```

The script defaults to the selected validation configuration:

```text
n_factors = 10
learning_rate = 0.02
regularization = 0.10
n_epochs = 60
```

## Results

Stage 05 was tuned on `ratings_valid.csv`, then checked once on
`ratings_test.csv` after the validation choice was made.

### Selected Matrix Factorization Model

```text
 split  rows     rmse      mae
 valid 20000 1.082666 0.857215
  test 20000 1.030065 0.828554
```

### Baseline Comparison

```text
 model                   valid_rmse  valid_mae  test_rmse  test_mae
 Stage 03 bias baseline    1.089829   0.858936   1.032749  0.824633
 Stage 04 item-CF          1.177426   0.966152   1.110922  0.931586
 Stage 05 matrix factors   1.082666   0.857215   1.030065  0.828554
```

Stage 05 becomes the strongest current model by RMSE on both validation and
test splits. The Stage 03 bias baseline still has slightly better test MAE,
which means matrix factorization reduces larger squared errors more effectively
but does not dominate every average-error metric.

### Tuning Notes

- `n_factors=10` was better than 5, 20, or 40 factors under the selected
  training setup.
- `learning_rate=0.02` learned faster and reached a better validation RMSE than
  0.005, 0.01, or 0.03.
- `n_epochs=60` was the best checked point before validation RMSE flattened or
  began to rise slightly.
- `regularization=0.10` was the best RMSE setting, while `0.08` had nearly
  identical MAE. RMSE remains the main model-selection metric for this stage.
