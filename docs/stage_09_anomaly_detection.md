# Stage 09 - Anomaly Detection

This stage adds anomaly detection for unusual user behavior.

The model idea is:

```text
Use behavior features to flag users whose rating patterns look unusual.
```

Stage 08 grouped users into broad segments. Stage 09 asks a different question:
which users do not fit normal behavior patterns?

## Goal

Build a repeatable anomaly detection workflow that:

1. Summarizes each user's rating behavior.
2. Builds anomaly-oriented features such as rating count, mean rating, and
   rating variability.
3. Fits an unsupervised anomaly detector.
4. Reports unusual users with interpretable feature values.
5. Explains how anomaly detection differs from clustering and supervised
   prediction.

## Learning Path

1. Build user-level anomaly features.
2. Understand why anomaly detection has no direct `rating` target.
3. Fit a simple unsupervised detector.
4. Interpret anomaly labels and anomaly scores.
5. Compare anomaly detection with clustering.

## Concepts

### Clustering vs Anomaly Detection

Stage 08 clustering tries to describe common groups:

```text
typical user behavior -> user segments
```

Stage 09 anomaly detection looks for rare or unusual patterns:

```text
typical user behavior -> normal users + unusual users
```

### Evaluation

MovieLens does not provide ground-truth anomaly labels. That means anomaly
detection needs careful interpretation rather than RMSE-style evaluation.
The first acceptance target is therefore a stable workflow and interpretable
feature output, not a single "best" accuracy score.

## Acceptance Checks

Stage 09 starts with a learning scaffold. Unskip and complete tests as each
piece is implemented:

```bash
python -m pytest tests/test_anomaly.py -v
python -m pytest -q
python -m ruff check .
```
