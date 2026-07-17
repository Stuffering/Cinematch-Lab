"""Train and evaluate simple rating prediction baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.baselines import (
    BiasBaseline,
    GlobalMeanBaseline,
    ItemMeanBaseline,
    UserMeanBaseline,
)
from cinematch.metrics import mae, rmse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Compare rating baselines on the validation split."""
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

    args = parser.parse_args()

    ratings_train = pd.read_csv(args.processed_directory / "ratings_train.csv")
    ratings_eval = pd.read_csv(
        args.processed_directory / f"ratings_{args.eval_split}.csv"
    )

    baselines = {
        "global_mean": GlobalMeanBaseline(),
        "user_mean": UserMeanBaseline(),
        "item_mean": ItemMeanBaseline(),
        "bias": BiasBaseline(),
    }

    rows: list[dict[str, float | str]] = []
    for name, model in baselines.items():
        model.fit(ratings_train)
        predictions = model.predict(ratings_eval)
        rows.append(
            {
                "model": name,
                "rmse": rmse(ratings_eval["rating"], predictions),
                "mae": mae(ratings_eval["rating"], predictions),
            }
        )

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    print(f"evaluation split: {args.eval_split}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
