import numpy as np


def accuracy(preds: np.ndarray, y: np.ndarray) -> float:
    """A default metric for classifier ensembles

    Parameters
    ----------
    preds : np.ndarray
        The predictions

    y : np.ndarray
        The targets

    Returns
    -------
    float
        The accuracy of how often the predictions matched the targets
    """
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.flatten()

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()

    if preds.shape != y.shape:
        raise ValueError("Preds and y should have matching shapes")

    if preds.ndim == 1:
        return sum(preds == y) / len(preds)

    elif preds.ndim == 2:
        return sum(np.equal(preds, y).all(axis=1).astype(bool)) / len(preds)

    else:
        raise NotImplementedError()


def rmse(preds: np.ndarray, y: np.ndarray) -> float:
    """A default metric for regression ensembles

    Parameters
    ----------
    preds : np.ndarray
        The predictions

    y : np.ndarray
        The targets

    Returns
    -------
    float
        The root-mean-squared error of between the predictions and the targets
    """
    scores = np.sqrt(np.average((y - preds) ** 2, axis=0))
    if y.ndim > 1:
        return np.average(scores)
    else:
        return scores
