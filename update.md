# CineMatch Lab

CineMatch Lab is an offline movie recommendation and user behavior analysis
project built with MovieLens 100K. It is developed in verified stages and will
combine supervised learning, clustering, anomaly detection, recommender
systems, neural networks, and reinforcement learning in one coherent product.

## Current Stage

Stage 9 added anomaly detection for unusual user behavior. Stage 8 added
unsupervised user clustering from rating behavior and user metadata. Stage 7
added supervised rating prediction from explicit user
and movie features plus training-only rating history aggregates. Stage 6 added
content-based recommendation from MovieLens
genre features. Stage 5 added matrix factorization,
moving from item-item similarity to learned user and movie latent factors
trained with SGD. Stage 4 added item-based collaborative filtering with
user-item matrices, centered rating vectors, cosine similarity, similar-movie
lookup, user-level recommendations, and item-CF rating evaluation.

## Environment Setup

The environment is stored inside this project so initialization does not write
project content elsewhere.

```bash
cd "/Users/qianjiangyue/Desktop/Study Zone/个人项目"
conda env create --prefix ./.conda/envs/cinematch-lab --file environment.yml
conda activate "$(pwd)/.conda/envs/cinematch-lab"
```

For later dependency changes:

```bash
conda env update --prefix ./.conda/envs/cinematch-lab --file environment.yml --prune
```

## Download Data

```bash
python scripts/download_movielens.py
```

The command is idempotent: when the expected files already exist, it exits
without replacing them. MovieLens data is generated under `data/raw` and is
not committed to Git.

## Verify and Run

```bash
python -c "import cinematch; print(cinematch.__version__)"
python -m pytest
python -m ruff check .
python scripts/find_similar_movies.py --item-id 50 --n 10
python scripts/recommend_movies.py --user-id 1 --n 10
python scripts/evaluate_item.py --eval-split valid
python -m pytest tests/test_matrix_factorization.py -v
python scripts/train_matrix_factorization.py --eval-split valid
python -m pytest tests/test_content.py -v
python scripts/recommend_content.py --user-id 1 --n 10
python -m pytest tests/test_supervised.py -v
python scripts/train_supervised.py --eval-split valid
python -m pytest tests/test_clustering.py -v
python scripts/cluster_users.py --n-clusters 4
python -m pytest tests/test_anomaly.py -v
python scripts/detect_anomalies.py --contamination 0.05 --n 20
python -m streamlit run app/main.py
```

The Streamlit application is then available at `http://localhost:8501`.

## Development Workflow

Each stage follows the same loop: read the referenced notes, implement the
student-owned core logic, run tests, review results, commit the completed stage,
and record meaningful decisions in `update.md`.

Development-only learning markers are removed before portfolio submission:

```bash
python scripts/check_submission.py
```

## Data Entities

Project modules use stable identifiers and fields: `user_id`, `item_id`,
`rating`, `timestamp`, and movie metadata. Dataset-specific parsing stays at
the data boundary.


## 0.7.0 - 2026-07-20

- Started Stage 07 supervised rating models.
- Added a supervised learning scaffold for joining ratings with user and movie
  metadata, splitting features from targets, and identifying preprocessing
  feature types.
- Added skipped tests for the next step-by-step supervised learning path.
- Added Stage 07 documentation connecting explicit features to rating
  regression and model comparison.

## 0.7.1 - 2026-07-20

- Completed the Stage 07 supervised rating model pipeline with Ridge
  regression.
- Added training-only rating history features for user and item rating counts
  plus mean ratings.
- Added numeric imputation, standard scaling, categorical imputation, and
  one-hot encoding in a scikit-learn pipeline.
- Tuned Ridge alpha on the validation split and selected alpha=1.0 based on
  RMSE.
- Final Stage 07 results:
  - Validation RMSE/MAE: 1.085629 / 0.862677
  - Test RMSE/MAE: 1.043925 / 0.843649
- Key learning: supervised models depend heavily on feature quality. Explicit
  metadata and historical aggregates produced a competitive model, while
  matrix factorization still performed better by learning latent user-item
  interactions.

## 0.8.0 - 2026-07-20

- Started Stage 08 user clustering.
- Added a clustering scaffold for building one row of user-level behavior
  features and assigning unsupervised user segments.
- Added skipped tests for the next step-by-step clustering learning path.
- Added Stage 08 documentation connecting supervised regression to
  unsupervised segmentation.
- Implemented user-level clustering features from rating behavior:
  - rating_count
  - mean_rating
- Joined user metadata so clustering profiles can include age and occupation.
- Added KMeans user segment assignment with standardized numeric features.
- Added silhouette scoring to compare clustering quality.
- Added a user clustering script that reports clustering features, silhouette
  score, segment profiles, and sample assignments.
- Compared 3, 4, and 5 clusters:
  - 3 clusters silhouette score: 0.300
  - 4 clusters silhouette score: 0.307
  - 5 clusters silhouette score: 0.280
- Selected 4 clusters as the Stage 08 default because it had the best
  silhouette score and produced interpretable user segments.

## 0.9.0 - 2026-07-20

- Started Stage 09 anomaly detection.
- Added an anomaly detection scaffold for building user-level anomaly features
  and flagging unusual user behavior.
- Added skipped tests for the next step-by-step anomaly detection learning
  path.
- Added Stage 09 documentation connecting clustering to anomaly detection.

## 0.9.1 - 2026-07-20

- Completed the Stage 09 anomaly detection workflow with IsolationForest.
- Implemented user-level anomaly features:
  - rating_count
  - mean_rating
  - rating_std
  - min_rating
  - max_rating
- Added standardized feature scaling before anomaly detection.
- Added a user anomaly detection script that reports the contamination setting,
  detected anomaly count, anomaly scores, and top anomalous users.
- Compared contamination settings:
  - 0.01 detected 6 anomalies out of 590 users
  - 0.05 detected 30 anomalies out of 590 users
  - 0.10 detected 59 anomalies out of 590 users
- Selected contamination=0.05 as the Stage 09 default because it captures a
  clear but not overly broad anomaly set.
- Key learning: anomaly_score provides a continuous ranking of unusual users,
  while contamination controls where that ranking is thresholded into
  is_anomaly labels.
