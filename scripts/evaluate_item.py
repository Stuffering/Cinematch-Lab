"""Evaluate item-based collaborative filtering rating predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.collaborative import predict_ratings_item_based
from cinematch.metrics import mae, rmse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Evaluate item-based collaborative filtering on a prepared split."""
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
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum item similarity used for prediction",
    )
    parser.add_argument(
        "--min-common-users",
        type=int,
        default=5,
        help="Minimum users who rated both target and source items",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional number of evaluation rows for quick local checks",
    )
    args = parser.parse_args()

    ratings_train = pd.read_csv(args.processed_directory / "ratings_train.csv")
    ratings_eval = pd.read_csv(
        args.processed_directory / f"ratings_{args.eval_split}.csv"
    )
    if args.max_rows is not None:
        ratings_eval = ratings_eval.head(args.max_rows)

    predictions = predict_ratings_item_based(
        train_ratings=ratings_train,
        rows_to_predict=ratings_eval,
        min_similarity=args.min_similarity,
        min_common_users=args.min_common_users,
    )

    result = {
        "model": "item_cf",
        "eval_split": args.eval_split,
        "rows": len(ratings_eval),
        "rmse": rmse(ratings_eval["rating"], predictions),
        "mae": mae(ratings_eval["rating"], predictions),
    }

    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
