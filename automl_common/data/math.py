from typing import Iterable

import numpy as np


def weighted_sum(
    weights: Iterable[float],
    arrays: Iterable[np.ndarray],
) -> np.ndarray:
    """Computes a weighted sum of arrays

    Tries to be performant over using a generic `sum` by storing into a single array

    Note:
    -----
    We do no validation of lengths, this is introduced in Python 3.10

    Parameters
    ----------
    weights: Iterable[float]
        The weights to apply

    arrays: Iterable[np.ndarray]
        The arrays to sum together

    Returns
    -------
    np.ndarray
        The weighted sum of the arrays
    """
    iterator = zip(iter(weights), iter(arrays))

    # Do the first one to get an outputnlocation
    weight, arr = next(iterator)
    buff = weight * arr.copy()

    for weight, arr in iterator:
        np.add(buff, weight * arr, out=buff)

    return buff
