"""Utilities for saving and loading model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Save a model artifact and create parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)

    return path


def load_model_artifact(path: Path) -> dict[str, Any]:
    """Load a model artifact saved with joblib."""
    return joblib.load(path)
