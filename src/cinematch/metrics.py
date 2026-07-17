"""Evaluation metrics for rating prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Compute root mean squared error."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    errors = true_values - predicted_values
    squared_errors = errors ** 2
    mean_squared_error = np.mean(squared_errors)

    return float(np.sqrt(mean_squared_error))


def mae(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Compute mean absolute error."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    errors = true_values - predicted_values
    absolute_errors = np.abs(errors)
    mean_absolute_error = np.mean(absolute_errors)

    return float(mean_absolute_error)
