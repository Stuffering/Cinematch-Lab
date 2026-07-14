"""Download and extract MovieLens 100K into the project data directory."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
EXPECTED_FILES = ("u.data", "u.item", "u.user", "u.genre", "README")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw"


def is_complete(dataset_directory: Path) -> bool:
    """Return whether the expected MovieLens files are already present."""
    return all((dataset_directory / name).is_file() for name in EXPECTED_FILES)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive after rejecting paths outside the destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zip_file.extractall(destination)


def download_dataset(output_directory: Path = DEFAULT_OUTPUT) -> Path:
    """Download MovieLens 100K once and return its extracted directory."""
    output_directory = output_directory.resolve()
    dataset_directory = output_directory / "ml-100k"
    if is_complete(dataset_directory):
        print(f"MovieLens 100K already exists: {dataset_directory}")
        return dataset_directory

    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary:
        temporary_directory = Path(temporary)
        archive = temporary_directory / "ml-100k.zip"
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, archive)
        safe_extract(archive, temporary_directory)
        extracted = temporary_directory / "ml-100k"
        if not is_complete(extracted):
            raise RuntimeError("Downloaded archive is missing expected MovieLens files")
        if dataset_directory.exists():
            shutil.rmtree(dataset_directory)
        shutil.move(str(extracted), dataset_directory)

    print(f"MovieLens 100K is ready: {dataset_directory}")
    return dataset_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Parent directory for the extracted ml-100k directory",
    )
    args = parser.parse_args()
    download_dataset(args.output)


if __name__ == "__main__":
    main()

