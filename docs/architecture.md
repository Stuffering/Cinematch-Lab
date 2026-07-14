# Architecture

CineMatch Lab uses a staged offline machine learning workflow:

```text
MovieLens data -> validation and features -> experiments and models
              -> evaluation artifacts -> Streamlit application
```

- `src/cinematch` contains reusable data, model, evaluation, and inference code.
- `scripts` contains repeatable command-line workflows.
- `notebooks` is limited to exploration; reusable logic must move into `src`.
- `app` reads prepared artifacts and does not train models during page rendering.
- generated datasets and model artifacts remain outside Git.

