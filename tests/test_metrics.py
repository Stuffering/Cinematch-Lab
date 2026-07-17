import numpy as np
import pytest

from cinematch.metrics import mae, rmse


def test_rmse_computes_root_mean_squared_error() -> None:
    y_true = np.array([5, 3, 1])
    y_pred = np.array([4, 3, 3])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))


def test_mae_computes_mean_absolute_error() -> None:
    y_true = np.array([5, 3, 1])
    y_pred = np.array([4, 3, 3])

    assert mae(y_true, y_pred) == pytest.approx(1.0)
