import numpy as np
import pandas as pd
from numpy.typing import NDArray


def linear_weighted_average(
    vector: NDArray,
    inc: bool = True,
    axis: int = 0,
) -> float:
    """
    Compute a linearly weighted average of a vector.

    Parameters
    ----------
    vector : NDArray
        Input values.
    inc : bool
        If True, weights increase with time. If False, decrease.
    axis : int
        Axis along which to average.

    Returns
    -------
    float
        Weighted average.
    """
    n = len(vector)
    if inc:
        weights = np.arange(1, n + 1)
    else:
        weights = np.arange(n, 0, -1)
    return np.average(vector, weights=weights, axis=axis)


def rmse(
    y_true: NDArray,
    y_pred: NDArray,
    weights: str | None = None,
    axis: int = 0,
) -> float:
    """
    Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : NDArray
        Ground-truth values.
    y_pred : NDArray
        Predicted values.
    weights : str or None
        If "linear", applies increasing linear weights.
    axis : int
        Axis along which to compute the error.

    Returns
    -------
    float
        RMSE value.
    """
    if weights == "linear":
        mse = linear_weighted_average((y_true - y_pred) ** 2, inc=True, axis=axis)
    else:
        mse = np.mean((y_true - y_pred) ** 2, axis=axis)
    return np.sqrt(mse)


def score(
    true: NDArray,
    pred: NDArray,
    a: float = 1 / 13,
    b: float = 1 / 10,
    axis: int = 0,
) -> float:
    """
    Asymmetric exponential scoring function (NASA-style).

    Penalizes late predictions more than early ones.

    Parameters
    ----------
    true : NDArray
        True RUL values.
    pred : NDArray
        Predicted RUL values.
    a, b : float
        Penalty coefficients.
    axis : int
        Axis along which to sum.

    Returns
    -------
    float
        Score value (lower is better).
    """
    delta = true - pred
    exp = np.exp(a * delta) * (delta > 0) + np.exp(-b * delta) * (delta <= 0) - 1
    return np.sum(exp, axis=axis)


def picp(
    true: NDArray,
    lower: NDArray,
    upper: NDArray,
    margin: float = 0.0,
    weights: str | None = None,
    axis: int = 0,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).

    Parameters
    ----------
    true : NDArray
        True values.
    lower, upper : NDArray
        Prediction interval bounds.
    margin : float
        Optional tolerance margin.
    weights : str or None
        If "linear", applies decreasing linear weights.
    axis : int
        Axis along which to compute coverage.

    Returns
    -------
    float
        Coverage probability in [0, 1].
    """
    captured = np.logical_and(lower - margin <= true, true <= upper + margin)
    if weights == "linear":
        return linear_weighted_average(captured, inc=False, axis=axis)
    return np.mean(captured, axis=axis)


def pinaw(
    lower: NDArray,
    upper: NDArray,
    range: float,
    weights: str | None = None,
    axis: int = 0,
) -> float:
    """
    Prediction Interval Normalized Average Width (PINAW).

    Parameters
    ----------
    lower, upper : NDArray
        Prediction interval bounds.
    range : float
        Range of the true target values.
    weights : str or None
        If "linear", applies increasing linear weights.
    axis : int
        Axis along which to average.

    Returns
    -------
    float
        Normalized interval width.
    """
    assert np.all(upper >= lower), "upper must be >= lower"
    width = upper - lower
    if weights == "linear":
        return linear_weighted_average(width, inc=True, axis=axis) / range
    return np.mean(width, axis=axis) / range


def build_metrics_table(
    df: pd.DataFrame,
    weights: str | None = None,
    picp_margin: float = 0.0,
) -> pd.DataFrame:
    """
    Compute RUL metrics per unit.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ['unit', 'true_rul', 'mean', 'lower', 'upper']
    weights : str or None
        Weighting scheme ("linear" or None).
    picp_margin : float
        Margin for PICP.

    Returns
    -------
    pd.DataFrame
        Metrics table (one row per unit).
    """
    rows = []

    for unit, g in df.groupby("unit"):
        true = g["true_rul"].to_numpy()
        mean = g["mean"].to_numpy()
        lower = g["lower"].to_numpy()
        upper = g["upper"].to_numpy()

        rows.append(
            {
                "unit": unit,
                "RMSE": rmse(true, mean, weights=weights),
                "Score": score(true, mean),
                "PICP": picp(true, lower, upper, margin=picp_margin, weights=weights),
                "PINAW": pinaw(lower, upper, range=true.max(), weights=weights),
            }
        )

    return pd.DataFrame(rows)
