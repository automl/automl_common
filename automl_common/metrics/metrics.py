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
    return (preds == y).sum() / len(preds)


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
