# CineMatch Lab

CineMatch Lab is an offline movie recommendation and user behavior analysis
project built with MovieLens 100K. It is developed in verified stages and will
combine supervised learning, clustering, anomaly detection, recommender
systems, neural networks, and reinforcement learning in one coherent product.

## Current Stage

Stage 12 is starting a recommendation agent interface that can validate
recommendation modes and prepare user requests for future tool orchestration.
Stage 11 added a hybrid recommendation system that blends multiple
recommendation sources. Stage 10 added neural rating prediction with
TensorFlow/Keras. Stage 9 added anomaly detection for unusual user behavior.
Stage 8 added unsupervised user clustering from rating behavior and user
metadata. Stage 7 added supervised rating prediction from explicit user
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
python -m pytest tests/test_neural.py -v
python scripts/train_neural.py --eval-split valid --epochs 20
python -m pytest tests/test_hybrid.py -v
python scripts/recommend_hybrid.py --user-id 1 --n 10
python -m pytest tests/test_recommendation_agent.py -v
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
