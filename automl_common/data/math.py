from typing import Iterable

import numpy as np


def weighted_sum(
    weights: Iterable[float],
    predictions: Iterable[np.ndarray],
) -> np.ndarray:
    """Computes a weighted sum of predictions

    Tries to be performant over using a generic `sum` by storing into a single array

    Parameters
    ----------
    weights: Iterable[float]
        The weights to apply

    predictions: Iterable[np.ndarray]
        The predictions to sum together

    Returns
    -------
    np.ndarray
        The weighted sum of the predictions
    """
    weight = next(iter(weights))
    pred = next(iter(predictions))

    buff = weight * pred.copy()

    for weight, pred in zip(weights, predictions):
        np.add(buff, weight * pred, out=buff)

    return buff
