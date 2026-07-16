"""Validate the local MovieLens 100K dataset."""

from __future__ import annotations

from pathlib import Path

from cinematch.data import load_dataset, validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "ml-100k"


def main() -> None:
    """Load MovieLens tables and run basic relationship validation."""
    ratings, users, movies = load_dataset(DEFAULT_DATASET)
    validate_dataset(ratings, users, movies)

    print(f"ratings: {ratings.shape}")
    print(f"users: {users.shape}")
    print(f"movies: {movies.shape}")
    print("validation passed")


if __name__ == "__main__":
    main()
