# Stage 12 - Recommendation Agent Interface

This stage starts a recommendation interface that can be used by a future
agentic product layer.

The idea is:

```text
User request -> recommendation mode decision -> recommender execution
```

Earlier stages built individual recommendation and modeling capabilities.
Stage 12 begins wrapping those capabilities behind a smaller decision interface
so future UI or agent code does not need to know every implementation detail.

## Goal

Build a repeatable recommendation interface that:

1. Lists supported recommendation modes.
2. Validates user-selected modes.
3. Converts user input into a structured recommendation request.
4. Chooses a sensible mode when the user asks for automatic selection.
5. Prepares the project for later model artifact loading and serving.

## Learning Path

1. Define supported recommendation modes.
2. Validate a selected recommendation mode.
3. Build a structured recommendation request.
4. Choose a recommendation mode from user context.
5. Connect the request to existing recommendation scripts.

## Concepts

### Why an Agent Interface?

The earlier stages answer different technical questions:

```text
item-CF: What did similar item behavior suggest?
content: What explicit movie features match the user?
hybrid: How can several recommendation sources be blended?
neural: What rating might a learned embedding model predict?
```

An agent interface answers a product question:

```text
Given the user's request and context, which recommendation path should run?
```

This is the beginning of tool orchestration. The agent layer should decide
which existing capability to use instead of rewriting the recommender logic.

### Model Artifacts

Some models are expensive to train repeatedly. Later stages can persist trained
artifacts such as embeddings, scikit-learn pipelines, and Keras models so the
recommendation interface can load them as options instead of retraining each
time.

## Acceptance Checks

Stage 12 starts with a learning scaffold. Unskip and complete tests as each
piece is implemented:

```bash
python -m pytest tests/test_recommendation_agent.py -v
python -m pytest tests/test_recommend_agent.py -v
python scripts/recommend_agent.py --user-id 1 --mode auto --n 10
python -m ruff check src/cinematch/recommendation_agent.py tests/test_recommendation_agent.py scripts/recommend_agent.py tests/test_recommend_agent.py
```
