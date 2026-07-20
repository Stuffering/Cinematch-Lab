"""Train and evaluate a neural rating prediction model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.metrics import mae, rmse
from cinematch.neural import (
    build_interaction_mappings,
    build_neural_rating_model,
    encode_interactions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Train a neural rating model and evaluate it on a prepared split."""
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
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for users and items",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional number of training rows for quick checks",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Optional number of evaluation rows for quick checks",
    )
    args = parser.parse_args()

    ratings_train = pd.read_csv(args.processed_directory / "ratings_train.csv")
    ratings_eval = pd.read_csv(
        args.processed_directory / f"ratings_{args.eval_split}.csv"
    )

    if args.max_train_rows is not None:
        ratings_train = ratings_train.head(args.max_train_rows)

    if args.max_eval_rows is not None:
        ratings_eval = ratings_eval.head(args.max_eval_rows)

    user_to_index, item_to_index = build_interaction_mappings(ratings_train)

    ratings_eval = ratings_eval[
        ratings_eval["user_id"].isin(user_to_index)
        & ratings_eval["item_id"].isin(item_to_index)
    ]

    x_train, y_train = encode_interactions(
        ratings_train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )
    x_eval, y_eval = encode_interactions(
        ratings_eval,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )

    model = build_neural_rating_model(
        n_users=len(user_to_index),
        n_items=len(item_to_index),
        embedding_dim=args.embedding_dim,
    )

    model.fit(
        {
            "user_index": x_train["user_index"],
            "item_index": x_train["item_index"],
        },
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )

    predictions = model.predict(
        {
            "user_index": x_eval["user_index"],
            "item_index": x_eval["item_index"],
        },
        verbose=0,
    ).reshape(-1)

    result = {
        "model": "neural_embedding",
        "eval_split": args.eval_split,
        "train_rows": len(x_train),
        "eval_rows": len(x_eval),
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "rmse": rmse(y_eval, predictions),
        "mae": mae(y_eval, predictions),
    }

    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()