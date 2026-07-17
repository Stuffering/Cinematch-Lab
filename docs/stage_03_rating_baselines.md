# Stage 03 - Rating Prediction Baselines

This stage builds the first modeling layer for CineMatch Lab. The goal is not
to create an advanced recommender yet, but to establish reliable evaluation
metrics and simple baselines that every later model must beat.

## Goal

Use the processed MovieLens rating splits from Stage 02 to evaluate simple
rating predictors.

The stage should answer:

- What does a simple global-average predictor achieve?
- Does using user history improve validation performance?
- Does using item history improve validation performance?
- Are RMSE and MAE implemented consistently and tested with small examples?
- Is validation used for model comparison while test remains reserved for final
  evaluation?

## Scope

Create reusable metric and baseline model modules, then expose them through a
repeatable training script.

This stage does not implement collaborative filtering, matrix factorization,
neural networks, or Streamlit integration. Those come later.

## Student-Owned Tasks

1. Implement `rmse(...)` in `src/cinematch/metrics.py`.
2. Implement `mae(...)` in `src/cinematch/metrics.py`.
3. Implement `GlobalMeanBaseline` in `src/cinematch/baselines.py`.
4. Implement `UserMeanBaseline` in `src/cinematch/baselines.py`.
5. Implement `ItemMeanBaseline` in `src/cinematch/baselines.py`.
6. Implement `BiasBaseline` in `src/cinematch/baselines.py`.
7. Unskip and complete metric tests in `tests/test_metrics.py`.
8. Unskip and complete baseline tests in `tests/test_baselines.py`.
9. Complete `scripts/train_baselines.py` to compare baselines on validation
   data.
10. Record validation results in this document before choosing the next model.

## Expected Baselines

### Global Mean

Predict every rating as the average rating in the training set.

This is the simplest possible numeric baseline:

```text
prediction = train.rating.mean()
```

### User Mean

Predict using each user's average training rating. For users not seen in
training, fall back to the global mean.

### Item Mean

Predict using each movie's average training rating. For movies not seen in
training, fall back to the global mean.

### Bias Baseline

Predict using the global mean plus regularized user and item bias terms:

```text
prediction = global_mean + user_bias + item_bias
```

This baseline captures both user scoring tendencies and movie popularity while
shrinking sparse users or movies toward the global mean.

## Metrics

### RMSE

Root mean squared error penalizes large errors more strongly:

```text
sqrt(mean((y_true - y_pred) ** 2))
```

### MAE

Mean absolute error is easier to interpret as average rating-point error:

```text
mean(abs(y_true - y_pred))
```

## Data Input

Run Stage 02 preparation first:

```bash
python scripts/prepare_data.py
```

Stage 03 reads:

```text
data/processed/ratings_train.csv
data/processed/ratings_valid.csv
data/processed/ratings_test.csv
```

The default split remains 60% train, 20% validation, and 20% test.

## References

REFERENCE: Model evaluation and train/CV/test workflow:
`/Users/qianjiangyue/Desktop/Study Zone/知识点汇总/Advanced Learning Algorithms/Evaluation and Diagnostics（模型评估与诊断）/Model Evaluation（模型评估）.md`

REFERENCE: Bias and variance thinking for comparing baselines:
`/Users/qianjiangyue/Desktop/Study Zone/知识点汇总/Advanced Learning Algorithms/Evaluation and Diagnostics（模型评估与诊断）/Bias and variance（偏差和方差）.md`

REFERENCE: Course practice with train/CV/test evaluation:
`/Users/qianjiangyue/Desktop/Study Zone/示例及练习/第二课 Advanced Learning Algorithms/week3/8.Practice Lab Advice for applying machine learning/C2_W3_Assignment.ipynb`

REFERENCE: Recommender-system rating prediction background:
`/Users/qianjiangyue/Desktop/Study Zone/知识点汇总/Unsupervised learning recommenders reinforcement learning/Recommender System（推荐系统）/Recommender Systems with Per-Item Features（基于物品特征的推荐系统）.md`

## Results

Baseline models were trained on `ratings_train.csv` and evaluated on the
chronological validation and test splits.

### Validation Split

```text
      model     rmse      mae
       bias 1.089829 0.858936
  item_mean 1.094652 0.863742
global_mean 1.187128 0.984395
  user_mean 1.190462 0.980399
```

### Test Split

```text
      model     rmse      mae
       bias 1.032749 0.824633
  item_mean 1.044119 0.833387
global_mean 1.117590 0.942058
  user_mean 1.119190 0.941031
```

### Interpretation

- `bias` is the strongest baseline on both validation and test splits.
- `bias` improves on `item_mean` by combining user-level and item-level
  offsets around the global mean.
- `user_mean` does not improve over `global_mean` under the current
  chronological split.
- Future models should first beat the validation RMSE of `1.089829` from the
  bias baseline before being considered useful.

## Acceptance Checks

Run these before committing the completed stage:

```bash
python scripts/prepare_data.py
python scripts/train_baselines.py --eval-split valid
python scripts/train_baselines.py --eval-split test
python -m pytest -q
python -m ruff check .
```

Before final portfolio submission, remove development markers such as
`TODO(student)`, `HINT`, and `REFERENCE` from production code.
