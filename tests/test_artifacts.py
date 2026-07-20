from cinematch.artifacts import load_model_artifact, save_model_artifact


def test_save_model_artifact_creates_parent_directories(tmp_path) -> None:
    artifact_path = tmp_path / "nested" / "model.joblib"
    artifact = {
        "model": "example-model",
        "metadata": {"rmse": 1.0},
    }

    saved_path = save_model_artifact(artifact, artifact_path)

    assert saved_path == artifact_path
    assert artifact_path.exists()


def test_load_model_artifact_returns_saved_payload(tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"
    artifact = {
        "model": "example-model",
        "metadata": {"alpha": 1.0},
        "feature_columns": ["age", "gender"],
    }

    save_model_artifact(artifact, artifact_path)
    loaded = load_model_artifact(artifact_path)

    assert loaded == artifact
