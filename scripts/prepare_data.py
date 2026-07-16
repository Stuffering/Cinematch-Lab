"""Prepare cleaned MovieLens data for modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from cinematch.clean import clean_dataset
from cinematch.data import load_dataset, validate_dataset
from cinematch.split import split_ratings_train_valid_test_by_time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "ml-100k"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Prepare cleaned and split MovieLens tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data-directory",
        type=Path,
        default=DEFAULT_RAW_DATASET,
        help="Directory containing the extracted MovieLens 100K files",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where processed CSV files will be written",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Fraction of ratings reserved for validation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of ratings reserved for testing",
    )
    args = parser.parse_args()

    ratings, users, movies = load_dataset(args.raw_data_directory)

    validate_dataset(ratings, users, movies)

    cleaned_ratings, cleaned_users, cleaned_movies = clean_dataset(
        ratings,
        users,
        movies,
    )

    ratings_train, ratings_valid, ratings_test = (
        split_ratings_train_valid_test_by_time(
            cleaned_ratings,
            valid_size=args.valid_size,
            test_size=args.test_size,
        )
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    ratings_train.to_csv(args.output_directory / "ratings_train.csv", index=False)
    ratings_valid.to_csv(args.output_directory / "ratings_valid.csv", index=False)
    ratings_test.to_csv(args.output_directory / "ratings_test.csv", index=False)
    cleaned_users.to_csv(args.output_directory / "users.csv", index=False)
    cleaned_movies.to_csv(args.output_directory / "movies.csv", index=False)

    print(f"ratings_train: {ratings_train.shape}")
    print(f"ratings_valid: {ratings_valid.shape}")
    print(f"ratings_test: {ratings_test.shape}")
    print(f"users: {cleaned_users.shape}")
    print(f"movies: {cleaned_movies.shape}")
    print(f"processed data written to: {args.output_directory}")


if __name__ == "__main__":
    main()
