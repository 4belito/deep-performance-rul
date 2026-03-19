import numpy as np
import pandas as pd
from numpy.typing import NDArray


def rmse(
    y_true: NDArray,
    y_pred: NDArray,
    axis: int = 0,
) -> float:
    """
    Root Mean Squared Error (RMSE).
    """

    mse = np.mean((y_true - y_pred) ** 2, axis=axis)
    return np.sqrt(mse)


def nasa_score(
    true: NDArray,
    pred: NDArray,
    a: float = 1 / 13,
    b: float = 1 / 10,
    axis: int = 0,
) -> float:
    """
    Asymmetric exponential scoring function (NASA-style).
    Penalizes late predictions more than early ones.
    """
    delta = true - pred
    exp = np.exp(a * delta) * (delta > 0) + np.exp(-b * delta) * (delta <= 0) - 1
    return np.sum(exp, axis=axis)


def picp(
    true: NDArray,
    lower: NDArray,
    upper: NDArray,
    axis: int = 0,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).
    """
    captured = np.logical_and(lower <= true, true <= upper)
    return np.mean(captured, axis=axis)


def pinaw(
    lower: NDArray,
    upper: NDArray,
    range: float,
    axis: int = 0,
) -> float:
    """
    Prediction Interval Normalized Average Width (PINAW).
    """
    assert np.all(upper >= lower), "upper must be >= lower"
    width = upper - lower
    return np.mean(width, axis=axis) / range


def build_metrics_table(
    df: pd.DataFrame,
) -> dict[str, float]:

    true = df["true_rul"].to_numpy()
    mean = df["mean"].to_numpy()
    lower = df["lower"].to_numpy()
    upper = df["upper"].to_numpy()

    metrics = {
        "RMSE": rmse(true, mean),
        "Score": nasa_score(true, mean),
        "PICP": picp(true, lower, upper),
        "PINAW": pinaw(lower, upper, range=true.max()),
    }
    return metrics
