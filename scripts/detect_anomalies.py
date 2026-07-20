"""Detect unusual users from rating behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cinematch.anomaly import (
    build_user_anomaly_features,
    detect_user_anomalies,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Detect anomalous users and print the highest-scoring cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-directory",
        type=Path,
        default=DEFAULT_PROCESSED_DIRECTORY,
        help="Directory containing prepared MovieLens CSV files",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected share of users to flag as anomalies; Stage 09 default is 0.05",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of top anomalous users to display",
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.processed_directory / "ratings_train.csv")

    features = build_user_anomaly_features(ratings)
    anomalies = detect_user_anomalies(
        features,
        contamination=args.contamination,
    )

    report = anomalies.join(features, on="user_id")
    report = report.sort_values("anomaly_score", ascending=False)

    print(f"Contamination setting: {args.contamination:.3f}")
    print(f"Detected anomalies: {report['is_anomaly'].sum()} / {len(report)}")
    print()
    print("Top anomalous users:")
    print(report.head(args.n).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
