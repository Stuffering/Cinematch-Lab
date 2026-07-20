# Stage 11 - Hybrid Recommendation System

This stage adds a hybrid recommendation layer.

The model idea is:

```text
Combine multiple recommendation sources into one ranked list.
```

Earlier stages built separate recommenders and predictors: item-based
collaborative filtering, content-based recommendation, matrix factorization,
supervised models, and neural embeddings. Stage 11 starts turning those separate
capabilities into a unified recommendation interface.

## Goal

Build a repeatable hybrid recommendation workflow that:

1. Standardizes recommendation outputs from different sources.
2. Preserves each recommendation source for explainability.
3. Blends scores with configurable source weights.
4. Produces a single ranked recommendation list.
5. Prepares the project for an agentic recommendation layer.

## Learning Path

1. Define a common recommendation schema.
2. Standardize source-specific recommendation outputs.
3. Blend multiple sources with weights.
4. Inspect how each source contributes to the final score.
5. Decide how a future agent should choose source weights.

## Concepts

### Why Hybrid Recommendation?

Each recommender has strengths:

```text
item-CF: similar-user behavior signal
content: explainable movie metadata signal
matrix factorization/neural: latent interaction signal
baseline: robust fallback signal
```

A hybrid layer can combine these signals instead of forcing the product to rely
on one model.

### Common Schema

Stage 11 starts with a simple common schema:

```text
movie_id
title
source
source_score
```

This keeps the first hybrid layer model-agnostic. Recommendation sources can be
added later as long as they produce the same columns.

## Acceptance Checks

Stage 11 starts with a learning scaffold. Unskip and complete tests as each
piece is implemented:

```bash
python -m pytest tests/test_hybrid.py -v
python scripts/recommend_hybrid.py --user-id 1 --n 10
python -m pytest -q
python -m ruff check .
```
