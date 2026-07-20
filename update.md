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

## 0.3.0 - 2026-07-16

- Started Stage 03 rating baseline framework.
- Added metric, baseline model, and training script skeletons for
  student-owned implementation.
- Added skipped metric and baseline tests that define the expected behavior for
  RMSE, MAE, global mean, user mean, and item mean baselines.

## 0.3.1 - 2026-07-16

- Implemented RMSE and MAE for rating prediction evaluation.
- Implemented global mean, user mean, and item mean rating baselines.
- Added a regularized user/item bias baseline as the strongest current baseline.
- Completed baseline training script with selectable validation or test
  evaluation.
- Recorded baseline results; `bias` is the strongest current baseline with
  validation RMSE of 1.089829 and test RMSE of 1.032749.

## 0.4.0 - 2026-07-17

- Started Stage 04 item-based collaborative filtering.
- Added the collaborative filtering learning scaffold with user-item matrix,
  user-centering, cosine similarity, item similarity matrix, and top-similar
  item TODOs.
- Added skipped tests that define the expected behavior for each Stage 04
  learning step.
- Added Stage 04 documentation to keep the next modeling phase explicit and
  incremental.

## 0.4.1 - 2026-07-17

- Completed the first item-based collaborative filtering utilities.
- Added real-data similar movie lookup with a minimum common-user overlap
  threshold to reduce noisy one-user matches.
- Added script tests for similar movie lookup output and unknown item handling.

## 0.4.2 - 2026-07-17

- Added item-based user recommendations from movies a user rated highly.
- Added a real-data recommendation script that prints unseen movie titles for a
  selected user.
- Documented the first recommendation score as a simple similarity-weighted
  ranking score rather than a fully evaluated rating prediction model.

## 0.4.3 - 2026-07-17

- Added item-based rating prediction for a specific user/movie pair.
- Tested prediction behavior for similar centered ratings and fallback to the
  user's mean rating when no reliable similarity evidence is available.
- Documented the prediction formula as user mean plus a similarity-weighted
  centered preference adjustment.


## 0.4.4 - 2026-07-17

- Added shared item-pair similarity caching during batch item-CF prediction.
- Added cold-start fallback behavior for batch validation rows missing from the
  training matrix.
- Added a quick evaluation mode with `--max-rows` for iterative local checks.
- Verified a 2,000-row validation sample with RMSE 1.095493 and MAE 0.897054.

## 0.4.5 - 2026-07-17

- Replaced incremental similarity caching with precomputed item similarity and
  common-user overlap matrices for faster batch evaluation.
- Verified full validation and test splits for item-based CF: validation RMSE
  1.177426 / MAE 0.966152 and test RMSE 1.110922 / MAE 0.931586.
- Confirmed item-based CF is evaluable but does not yet beat the Stage 03
  regularized bias baseline.

## 0.4.6 - 2026-07-17

- Ran an item-CF parameter sweep over minimum common-user overlap and minimum
  similarity thresholds.
- Found the best validation setting at `min_common_users=1` and
  `min_similarity=0.05`, with validation RMSE 1.175697 and MAE 0.966276.
- Kept the default evaluation setting because the validation improvement did
  not hold on the test split, where the default remained slightly better.
- Cleaned remaining learning-marker comments from the EDA helper module before
  closing Stage 04.

## 0.5.0 - 2026-07-17

- Started Stage 05 matrix factorization.
- Added a latent-factor model scaffold with configuration, learned-parameter
  placeholders, and explicit `fit` / `predict` TODOs.
- Added skipped tests for the next learning steps: ID mappings, prediction
  alignment, and SGD training behavior.
- Added Stage 05 documentation connecting latent factors, dot products, loss
  functions, regularization, and SGD to the earlier baseline and item-CF work.

## 0.5.1 - 2026-07-20

- Completed the matrix factorization model with SGD training, fitted user/item
  biases, learned latent factors, clipped rating predictions, and cold-start
  fallback behavior.
- Added a repeatable matrix factorization training script with configurable
  factor count, learning rate, regularization, epochs, and quick-check row
  limits.
- Tuned Stage 05 on the validation split and selected `n_factors=10`,
  `learning_rate=0.02`, `regularization=0.10`, and `n_epochs=60`.
- Verified the final matrix factorization model on the test split with RMSE
  1.030065 and MAE 0.828554.
