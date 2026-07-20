"""Train and evaluate supervised rating prediction models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cinematch.artifacts import save_model_artifact
from cinematch.metrics import mae, rmse
from cinematch.supervised import (
    add_rating_history_features,
    build_preprocessing_feature_lists,
    build_rating_feature_table,
    split_features_and_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Train a supervised regression model and evaluate it on a prepared split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory containing prepared MovieLens CSV files",
    )
    parser.add_argument(
        "--eval-split",
        choices=["valid", "test"],
        default="valid",
        help="Prepared split used for evaluation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional number of training rows for quick local checks",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Optional number of evaluation rows for quick local checks",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Optional path where the trained model artifact will be saved",
    )
    args = parser.parse_args()

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

    train_table = build_rating_feature_table(
        ratings=ratings_train,
        users=users,
        movies=movies,
    )
    eval_table = build_rating_feature_table(
        ratings=ratings_eval,
        users=users,
        movies=movies,
    )

    train_table = add_rating_history_features(
        feature_table=train_table,
        train_ratings=ratings_train,
    )
    eval_table = add_rating_history_features(
        feature_table=eval_table,
        train_ratings=ratings_train,
    )

    x_train, y_train = split_features_and_target(train_table)
    x_eval, y_eval = split_features_and_target(eval_table)

    numeric_features, categorical_features = build_preprocessing_feature_lists(
        x_train
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=args.alpha)),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)

    result = {
        "model": "ridge_supervised",
        "eval_split": args.eval_split,
        "train_rows": len(train_table),
        "eval_rows": len(eval_table),
        "alpha": args.alpha,
        "rmse": rmse(y_eval, predictions),
        "mae": mae(y_eval, predictions),
    }

    print(pd.DataFrame([result]).to_string(index=False))

    if args.model_output is not None:
        artifact = {
            "model": model,
            "metadata": result,
            "feature_columns": x_train.columns.tolist(),
            "target_column": "rating",
        }
        artifact_path = save_model_artifact(
            artifact=artifact,
            path=args.model_output,
        )
        print(f"Saved model artifact to {artifact_path}")


if __name__ == "__main__":
    main()
