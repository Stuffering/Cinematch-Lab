# Stage 01 - Project Initialization

This development note records the initial project setup for CineMatch Lab. The
stage establishes a reproducible engineering foundation before any recommender
algorithm is implemented.

## Goal

Create an independent, portfolio-ready Python project for offline MovieLens
100K recommendation experiments and a later Streamlit application.

The initialization stage should answer:

- Can the project be recreated from Git and `environment.yml`?
- Can MovieLens 100K be downloaded without copying course data?
- Can the package be imported from `src`?
- Can tests, linting, and the minimal Streamlit app run?
- Are raw data, generated data, environments, and model artifacts excluded from
  Git?

## Scope

Stage 01 is engineering setup only. It intentionally avoids data analysis,
model training, recommendation algorithms, and final application features.

## Completed Tasks

1. Initialized an independent Git repository inside the project directory.
2. Created the staged project structure for application code, configs, data,
   docs, scripts, source modules, tests, reports, and models.
3. Added Conda environment configuration with Python 3.11 and project
   dependencies.
4. Added Python package metadata and test/lint configuration in
   `pyproject.toml`.
5. Added an idempotent MovieLens 100K downloader.
6. Added a minimal Streamlit entry point.
7. Added package import and downloader tests.
8. Added `.gitignore` rules for environments, caches, raw data, processed data,
   notebooks, model artifacts, and generated reports.
9. Added project overview, architecture notes, and development log.

## Key Files

- `README.md`: project overview, environment setup, data download, and run
  commands.
- `environment.yml`: Conda environment specification.
- `pyproject.toml`: package, pytest, and Ruff configuration.
- `scripts/download_movielens.py`: reproducible MovieLens 100K download.
- `scripts/check_submission.py`: final marker check before portfolio
  submission.
- `app/main.py`: minimal Streamlit application.
- `src/cinematch/__init__.py`: importable package boundary.
- `tests/test_package.py`: package import test.
- `tests/test_download.py`: downloader helper tests.

## Acceptance Checks

Run these checks from the project root:

```bash
python -c "import cinematch; print(cinematch.__version__)"
python scripts/download_movielens.py
python -m pytest -q
python -m ruff check .
python -m streamlit run app/main.py
```

The Streamlit command should open a local page, and MovieLens files should
remain under `data/raw/` without being committed to Git.

## Decisions

- Use MovieLens 100K as the fixed offline dataset.
- Use Conda with Python 3.11 for reproducibility.
- Keep generated datasets and model artifacts out of Git.
- Keep notebooks as local exploration only; reusable logic belongs in `src`.
- Use staged commits so each project phase is reviewable and reversible.
