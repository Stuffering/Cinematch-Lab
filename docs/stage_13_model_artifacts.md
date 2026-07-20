# Stage 13 - Model Artifacts

This stage adds a minimal model artifact workflow.

The idea is:

```text
Train model -> save artifact -> reuse artifact later
```

Earlier stages trained models directly inside scripts. That is fine for
learning the modeling logic, but repeated training is slow and unnecessary
when the model parameters are already available. Stage 13 starts separating
training from later reuse.

## Goal

Build a simple artifact workflow that:

1. Trains the supervised Ridge rating model.
2. Saves the fitted scikit-learn pipeline with joblib.
3. Stores useful metadata beside the model.
4. Loads the saved pipeline for later evaluation without retraining.
5. Keeps generated artifact files out of Git.

## Saved Payload

The supervised model artifact stores:

```text
model           fitted sklearn Pipeline
metadata        evaluation split, row counts, alpha, RMSE, MAE
feature_columns columns used by the model
target_column   rating
```

The artifact does not store the raw training data.

## Why This Matters

This step is mostly engineering glue, but it is still useful for ML project
acceptance:

```text
Training code proves the model can be learned.
Artifact code proves the learned model can be reused.
```

The first reuse path is `scripts/evaluate_supervised_artifact.py`, which loads
the saved pipeline and evaluates it on a prepared validation or test split
without fitting the model again.

## Acceptance Checks

```bash
python -m pytest tests/test_artifacts.py -v
python -m ruff check src/cinematch/artifacts.py tests/test_artifacts.py scripts/train_supervised.py scripts/evaluate_supervised_artifact.py
python scripts/train_supervised.py --eval-split valid --alpha 1.0 --max-train-rows 5000 --max-eval-rows 1000 --model-output models/supervised_ridge_sample.joblib
python scripts/evaluate_supervised_artifact.py --artifact-path models/supervised_ridge_sample.joblib --eval-split valid --max-train-rows 5000 --max-eval-rows 1000
```
