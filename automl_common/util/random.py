from typing import Union

import numbers

import numpy as np


def as_random_state(
    seed: Union[int, np.random.RandomState, None]
) -> np.random.RandomState:
    """Converts a valid seed arg into a numpy.random.RandomState

    Following convention of sklearn.

    Parameters
    ----------
    seed: Union[int, RandomState, None]
        The seed

    Returns
    -------
    RandomState
        A valid RandomState object to use
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError(f"Can't use {seed} to seed a numpy.random.RandomState instance")
