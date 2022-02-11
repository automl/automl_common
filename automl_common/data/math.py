from typing import Iterable, Optional

import numpy as np


def weighted_sum(
    arrays: Iterable[np.ndarray],
    weights: np.ndarray,
    dtype: Optional[np.dtype] = None,
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

    dtype: Optional[np.dtype] = None
        The dtype to store as. If none, uses the type of the first prediction

    Returns
    -------
    np.ndarray
        The weighted sum of the arrays
    """
    iterator = zip(weights, iter(arrays))

    # Do the first one to get an output location
    weight, arr = next(iterator)

    dtype = dtype if dtype is not None else arr.dtype
    buff = weight * arr.copy().astype(dtype)

    for weight, arr in iterator:
        np.add(buff, weight * arr.astype(dtype), out=buff)

    return buff


def normalize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalizes an array along an axis

    ..code:: python

        x = np.ndarray([
            [1, 1, 1],
            [2, 2, 2],
            [7, 7, 7],
        ])

        print(normalize(x, axis=0))

        np.ndarray([
            [.1, .1, .1]
            [.2, .2, .2]
            [.7, .7, .7]
        ])

        print(normalize(x, axis=1))

        np.ndarray([
            [.333, .333, .333]
            [.333, .333, .333]
            [.333, .333, .333]
        ])

    Note
    ----
    Does not account for 0 sums along an axis

    Parameters
    ----------
    x : np.ndarray
        The array to normalize

    axis : int = 0
        The axis to normalize across

    Returns
    -------
    np.ndarray
        The normalized array
    """
    return x / x.sum(axis=axis, keepdims=True)


def majority_vote(
    arrays: Iterable[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """ Do a majority vote where the weights correspond to each array's vote strength

    Parameters
    ----------
    arrays : Iterable[np.ndarray], (n, m)
        A iterable of `n` voters, each with `m` items in their array

    weights : Optional[np.ndarray] = None, (n,)
        `n` weights, the relative strength of each voter

    Returns
    -------
    np.ndarray (m,)
        An `m` long array with the most voted for item between all voted

    Raises
    ------
    ValueError
        If the number of weights and number of arrays don't match
    """
    # Each column is a voter votes, the row are the votes for a single item
    # v1   v2   v3
    # "a", "b", "a"   item 1
    # "b", "b", "c"   item 2
    # "c", "c", "b"   item 3
    # "a", "a", "b"   item 4
    arrays = list(arrays)
    arrs = np.array(arrays).T

    # weights, for voters v1, v2 and v3
    #  v1   v2   v3
    # 0.1, 0.2, 0.7
    _weights = np.ones(arrs.shape[0]) if weights is None else weights

    if len(_weights) != arrs.shape[1]:
        raise ValueError(
            f"`weights` ({len(_weights)}) and `arrays` ({arrs.shape[1]}) must have the same length."
        )

    # labels
    #  0     1    2
    # ["a", "b", "c"]
    labels = np.unique(arrs)

    # For eaxmple, first row
    # [
    #   weights[row == "a"].sum() -> weights[True, False, True].sum() -> [0.1, 0.7].sum() -> 0.8,
    #   weights[row == "b"].sum() -> ... -> 0.2
    #   weights[row == "c"].sum() -> ... -> 0.0
    # ]
    # == [0.8, 0.2, 0.0], meaning "a" has 0.8 voting strength, "b" has 0.2, and "c" has 0.0
    def _weighted_choice(row: np.ndarray) -> np.ndarray:
        return np.array([_weights[row == label].sum() for label in labels])

    # Perform this for each row
    weighted_choices = np.apply_along_axis(_weighted_choice, axis=1, arr=arrs)

    # Get the arg max of each row
    chosen_labels = np.argmax(weighted_choices, axis=1)

    # Map it back to the labels
    return labels[chosen_labels]
