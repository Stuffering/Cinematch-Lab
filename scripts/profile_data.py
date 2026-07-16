"""Print an exploratory data profile for the local MovieLens 100K dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cinematch.data import load_dataset, validate_dataset
from cinematch.eda import build_data_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "ml-100k"


def to_jsonable(value: Any) -> Any:
    """Convert profile values into JSON-friendly Python objects."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    """Load MovieLens tables and print their EDA profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DATASET,
        help="Directory containing the extracted MovieLens 100K files",
    )
    args = parser.parse_args()

    ratings, users, movies = load_dataset(args.data_directory)
    validate_dataset(ratings, users, movies)
    profile = build_data_profile(ratings, users, movies)

    print(json.dumps(profile, indent=2, default=to_jsonable))


if __name__ == "__main__":
    main()
