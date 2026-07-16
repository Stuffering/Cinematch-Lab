# CineMatch Lab Development Log

## 0.1.0 - 2026-07-14

- Initialized CineMatch Lab as an independent Git project.
- Chose MovieLens 100K for a reproducible offline portfolio project.
- Selected Python 3.11, Conda, pytest, TensorFlow, and Streamlit.
- Kept the Conda environment inside the project boundary to avoid modifying
  files elsewhere on the machine.
- Added an idempotent dataset downloader, minimal application, package import
  test, submission marker check, and initial architecture documentation.
- Deferred all model implementation to later student-owned stages.

## 0.2.0 - 2026-07-16

- Added MovieLens 100K data loading utilities for ratings, users, and movies.
- Added dataset validation for required columns, rating range, user references,
  and movie references.
- Added unit tests for valid tables and common invalid data cases.
- Added `scripts/validate_data.py` as a command-line validation entry point.
- Kept notebooks as local learning drafts and ignored them from Git.

## 0.2.1 - 2026-07-16

- Added EDA helpers for ratings, users, movies, and combined data profiling.
- Added unit tests for table-level EDA summaries and combined profile output.
- Added `scripts/profile_data.py` to print a JSON profile for the local
  MovieLens 100K dataset.
- Verified the profile script on the real dataset: 100,000 ratings, 943 users,
  and 1,682 movies.

## 0.2.2 - 2026-07-16

- Added reusable cleaning helpers for ratings, users, movies, and combined
  MovieLens tables.
- Added chronological rating split helpers to prevent future data leakage.
- Fixed the default modeling split at 60% train, 20% validation, and 20% test.
- Added `scripts/prepare_data.py` to regenerate processed CSV files under
  `data/processed/`.

## 0.2.3 - 2026-07-16

- Added `docs/stage_01_initialization.md` so stage documentation starts from
  project initialization.
- Updated `README.md` to reflect the current Stage 2 data preparation status.
