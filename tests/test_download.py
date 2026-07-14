from pathlib import Path

from scripts.download_movielens import is_complete


def test_complete_dataset_requires_all_expected_files(tmp_path: Path) -> None:
    dataset = tmp_path / "ml-100k"
    dataset.mkdir()
    assert not is_complete(dataset)

    for filename in ("u.data", "u.item", "u.user", "u.genre", "README"):
        (dataset / filename).touch()

    assert is_complete(dataset)

