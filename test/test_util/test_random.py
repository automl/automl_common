from typing import Union

import numpy as np
from pytest_cases import parametrize

from automl_common.util.random import as_random_state


@parametrize("seed", [None, 0, np.random.RandomState(5)])
def test_as_random_state(seed: Union[int, np.random.RandomState, None]) -> None:
    """
    Parameters
    ----------
    seed: Union[int, np.random.RandomState, None]
        The seed to convert to randomstate

    Expects
    -------
    * Should return a random state object
    """
    rng = as_random_state(seed)
    assert isinstance(rng, np.random.RandomState)
