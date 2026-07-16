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
