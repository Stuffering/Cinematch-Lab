"""Evaluate a saved supervised rating model artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.artifacts import load_model_artifact
from cinematch.metrics import mae, rmse
from cinematch.supervised import (
    add_rating_history_features,
    build_rating_feature_table,
    split_features_and_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "supervised_ridge.joblib"


def main() -> None:
    """Load a saved supervised model artifact and evaluate it on a split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory containing prepared MovieLens CSV files",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to a saved supervised model artifact",
    )
    parser.add_argument(
        "--eval-split",
        choices=["valid", "test"],
        default="valid",
        help="Prepared split used for evaluation",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional number of training rows used for matching history features",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Optional number of evaluation rows for quick local checks",
    )
    args = parser.parse_args()

    artifact = load_model_artifact(args.artifact_path)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})

    ratings_train = pd.read_csv(args.processed_directory / "ratings_train.csv")
    ratings_eval = pd.read_csv(
        args.processed_directory / f"ratings_{args.eval_split}.csv"
    )
    users = pd.read_csv(args.processed_directory / "users.csv")
    movies = pd.read_csv(args.processed_directory / "movies.csv")

    if args.max_train_rows is not None:
        ratings_train = ratings_train.head(args.max_train_rows)

    if args.max_eval_rows is not None:
        ratings_eval = ratings_eval.head(args.max_eval_rows)

    eval_table = build_rating_feature_table(
        ratings=ratings_eval,
        users=users,
        movies=movies,
    )
    eval_table = add_rating_history_features(
        feature_table=eval_table,
        train_ratings=ratings_train,
    )
    x_eval, y_eval = split_features_and_target(eval_table)
    predictions = model.predict(x_eval)

    result = {
        "artifact": str(args.artifact_path),
        "model": metadata.get("model", "unknown"),
        "eval_split": args.eval_split,
        "eval_rows": len(eval_table),
        "rmse": rmse(y_eval, predictions),
        "mae": mae(y_eval, predictions),
    }

    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
