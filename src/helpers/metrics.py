import numpy as np


def linear_weighted_average(vector, inc=True, axis=0):
    n = len(vector)
    if inc:
        weights = np.arange(1, n + 1)  # Linearly increasing weights
    else:
        weights = np.arange(n, 0, -1)
    return np.average(vector, weights=weights, axis=axis)


def rmse(y_true, y_pred, weights=None, axis=0):
    """RMSE: Root Mean Squared Error"""
    if weights == "linear":
        mse = linear_weighted_average((y_true - y_pred) ** 2, inc=True, axis=axis)
    else:
        mse = np.mean((y_true - y_pred) ** 2, axis=axis)
    return np.sqrt(mse)


def score(true, pred, a=1 / 13, b=1 / 10, axis=0):
    delta = true - pred
    exp = np.exp(a * delta) * (delta > 0) + np.exp(-b * delta) * (delta <= 0) - 1
    return np.sum(exp, axis=axis)


def picp(true, lower, upper, margin=0, weights=None, axis=0):
    """PICP: Prediction Interval Coverage Probability"""
    true_captured = np.logical_and(lower - margin <= true, true <= upper + margin)
    if weights == "linear":
        picp = linear_weighted_average(true_captured, inc=False, axis=axis)
    else:
        picp = np.mean(true_captured, axis=axis)
    return picp


def pinaw(lower, upper, range, weights=None, axis=0):
    """
    PINAW: Prediction Interval Normalized Averaged Width
    range: Range of the target variable over the forecast period(real value)
    """
    assert np.all(upper - lower) >= 0, "lower>=upper"
    width = upper - lower
    if weights == "linear":
        pinaw = linear_weighted_average(width, inc=True, axis=axis) / range
    else:
        pinaw = np.mean(width, axis=axis) / range
    return pinaw


def evaluate(true, preds, picp_marign=0, weights=None, axis=0):
    lower, mean, upper = preds.T
    RMSE = rmse(true, mean, weights=weights, axis=axis)
    Score = score(true, lower, axis=axis)
    PICP = picp(true, lower, upper, margin=picp_marign, weights=weights, axis=axis)
    PINAW = pinaw(lower, upper, range=np.max(true), weights=weights, axis=axis)
    return RMSE, Score, PICP, PINAW


def evaluate_dataset(trues, preds, picp_marign=0, weights=None, t_last=0):
    metrics = []
    for true, pred in zip(trues, preds):
        # add repetition dimension
        if t_last > 0:
            true = true[-t_last:]
            pred = pred[..., -t_last:, :]
        metrics.append(np.array(evaluate(true, pred, picp_marign=picp_marign, weights=weights)))
    metrics = np.stack(metrics)
    return np.mean(metrics, axis=(0, 2))
