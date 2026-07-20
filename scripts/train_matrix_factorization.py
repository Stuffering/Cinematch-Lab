"""Train and evaluate matrix factorization rating predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.matrix_factorization import MatrixFactorizationModel
from cinematch.metrics import mae, rmse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Train matrix factorization and evaluate it on a prepared split."""
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
        "--n-factors",
        type=int,
        default=10,
        help="Number of latent factors",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="SGD learning rate",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.1,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=60,
        help="Number of SGD passes over the training data",
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
    args = parser.parse_args()

    ratings_train = pd.read_csv(args.processed_directory / "ratings_train.csv")
    ratings_eval = pd.read_csv(
        args.processed_directory / f"ratings_{args.eval_split}.csv"
    )

    if args.max_train_rows is not None:
        ratings_train = ratings_train.head(args.max_train_rows)

    if args.max_eval_rows is not None:
        ratings_eval = ratings_eval.head(args.max_eval_rows)

    model = MatrixFactorizationModel(
        n_factors=args.n_factors,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        n_epochs=args.n_epochs,
        random_state=42,
    ).fit(ratings_train)

    predictions = model.predict(ratings_eval)

    result = {
        "model": "matrix_factorization",
        "eval_split": args.eval_split,
        "train_rows": len(ratings_train),
        "eval_rows": len(ratings_eval),
        "n_factors": args.n_factors,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "n_epochs": args.n_epochs,
        "rmse": rmse(ratings_eval["rating"], predictions),
        "mae": mae(ratings_eval["rating"], predictions),
    }

    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()