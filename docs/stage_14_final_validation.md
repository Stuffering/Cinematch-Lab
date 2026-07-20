# Stage 14 - Final Validation

This stage packages CineMatch Lab for final project review.

The goal is not to add another model. The goal is to verify that the complete
project is healthy, explainable, and easy to demonstrate.

## Validation Commands

```bash
python -m pytest
python -m ruff check .
python scripts/check_submission.py
```

Latest local validation result:

```text
81 passed
All checks passed
No development-only learning markers found.
```

The pytest run reports third-party deprecation warnings from matplotlib /
pyparsing. These warnings come from installed dependencies and do not indicate
project test failures.

## Demonstration Path

A compact review flow:

1. Show the README project scope and selected results.
2. Run package validation and tests.
3. Demonstrate item-based recommendations.
4. Demonstrate content-based recommendations.
5. Show supervised and neural evaluation results.
6. Demonstrate clustering and anomaly detection outputs.
7. Run hybrid recommendation and strategy routing.
8. Save and reload a supervised model artifact.

## Project Takeaway

CineMatch Lab now demonstrates both modeling breadth and engineering closure:

```text
data preparation -> models -> evaluation -> recommendations -> strategy routing -> artifact reuse
```

This is the final packaging layer for the current machine learning acceptance
scope.
