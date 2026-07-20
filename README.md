# CineMatch Lab

CineMatch Lab is an offline movie recommendation and user behavior analysis
project built with MovieLens 100K. It combines classical recommender systems,
supervised learning, clustering, anomaly detection, neural embeddings, hybrid
recommendation routing, and model artifact persistence in one verified
workflow.

## Project Scope

CineMatch Lab covers the full learning path from data preparation to reusable
model artifacts:

- Data ingestion, cleaning, validation, EDA, and train/validation/test splits
- Rating baselines and error metrics
- Item-based collaborative filtering and user-level recommendations
- Matrix factorization trained with SGD
- Content-based recommendation from explicit movie genre features
- Supervised Ridge rating prediction with metadata and rating-history features
- User clustering and anomaly detection from behavior summaries
- TensorFlow/Keras neural embedding rating prediction
- Hybrid recommendation blending and recommendation strategy routing
- Joblib model artifacts for saving and reusing trained supervised pipelines

## Selected Results

Representative local validation results:

- Supervised Ridge: valid RMSE/MAE `1.085629 / 0.862677`; test RMSE/MAE
  `1.043925 / 0.843649`
- Neural embedding model: known-subset valid RMSE/MAE approximately
  `1.02 / 0.805`; known-subset test RMSE/MAE approximately `1.06 / 0.847`
- User clustering: 4 clusters selected with silhouette score `0.307`
- Anomaly detection: contamination `0.05` detected `30 / 590` anomalous users
- Final validation: `81` tests passed, Ruff passed, and submission marker check
  passed

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

Core validation:

```bash
python -c "import cinematch; print(cinematch.__version__)"
python -m pytest
python -m ruff check .
python scripts/check_submission.py
```

Common workflow commands:

```bash
python scripts/find_similar_movies.py --item-id 50 --n 10
python scripts/recommend_movies.py --user-id 1 --n 10
python scripts/evaluate_item.py --eval-split valid
python -m pytest tests/test_matrix_factorization.py -v
python scripts/train_matrix_factorization.py --eval-split valid
python -m pytest tests/test_content.py -v
python scripts/recommend_content.py --user-id 1 --n 10
python -m pytest tests/test_supervised.py -v
python scripts/train_supervised.py --eval-split valid
python scripts/train_supervised.py --eval-split valid --alpha 1.0 --model-output models/supervised_ridge.joblib
python scripts/evaluate_supervised_artifact.py --artifact-path models/supervised_ridge.joblib --eval-split valid
python -m pytest tests/test_clustering.py -v
python scripts/cluster_users.py --n-clusters 4
python -m pytest tests/test_anomaly.py -v
python scripts/detect_anomalies.py --contamination 0.05 --n 20
python -m pytest tests/test_neural.py -v
python scripts/train_neural.py --eval-split valid --epochs 20
python -m pytest tests/test_hybrid.py -v
python scripts/recommend_hybrid.py --user-id 1 --n 10
python -m pytest tests/test_recommendation_strategy.py -v
python -m pytest tests/test_recommend_strategy.py -v
python scripts/recommend_strategy.py --user-id 1 --mode auto --n 10
python -m pytest tests/test_artifacts.py -v
python -m streamlit run app/main.py
```

The Streamlit application is then available at `http://localhost:8501`.

## Model Artifacts

Generated model files are written under `models/` and ignored by Git. Example:

```bash
python scripts/train_supervised.py --eval-split valid --alpha 1.0 --model-output models/supervised_ridge.joblib
python scripts/evaluate_supervised_artifact.py --artifact-path models/supervised_ridge.joblib --eval-split valid
```

The saved supervised artifact contains the fitted sklearn pipeline, evaluation
metadata, feature columns, and target column.

## Demonstration Flow

A concise project walkthrough:

1. Validate data and package health.
2. Show item-CF and content-based recommendations.
3. Compare supervised and neural rating prediction results.
4. Show clustering and anomaly summaries.
5. Run hybrid recommendations.
6. Run strategy-based recommendation routing.
7. Save and reload a supervised model artifact.

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
