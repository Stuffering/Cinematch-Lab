# Stage 10 - Neural Rating Models

This stage adds neural network rating prediction with TensorFlow/Keras.

The model idea is:

```text
Learn user and item embeddings inside a neural model, then predict ratings.
```

Stage 05 implemented matrix factorization manually with SGD. Stage 10 revisits
the same user-item prediction problem with a deep learning framework so the
project can connect classic recommender systems to modern neural modeling.

## Goal

Build a repeatable neural rating prediction workflow that:

1. Converts raw `user_id` and `item_id` values into contiguous integer indices.
2. Builds trainable user and item embedding layers.
3. Combines embeddings to predict ratings.
4. Trains on `ratings_train.csv`.
5. Evaluates on validation and test splits with RMSE and MAE.
6. Compares neural results with matrix factorization and supervised baselines.

## Learning Path

1. Build user and item ID mappings.
2. Encode rating interactions for neural model inputs.
3. Build a minimal Keras model with user and item embeddings.
4. Train on a small sample to confirm the pipeline.
5. Train on the full training split.
6. Compare validation/test metrics against previous stages.

## Concepts

### Embeddings

An embedding layer turns a discrete ID into a trainable vector:

```text
user_id -> user vector
item_id -> item vector
```

This is closely related to Stage 05 matrix factorization. The difference is
that TensorFlow manages the parameters, gradients, optimizer, and training loop.

### Neural Model vs Manual Matrix Factorization

Stage 05 explicitly updated user and item vectors with custom SGD logic. Stage
10 uses Keras layers and optimizers to express the model declaratively:

```text
inputs -> embedding layers -> interaction representation -> rating prediction
```

The first neural model should stay intentionally small. The purpose is to learn
the workflow before adding hidden layers or complex architecture.

## Acceptance Checks

Stage 10 starts with a learning scaffold. Unskip and complete tests as each
piece is implemented:

```bash
python -m pytest tests/test_neural.py -v
python -m pytest -q
python -m ruff check .
```
