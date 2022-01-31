from typing import Any, Union

import numpy as np
import pytest
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


@parametrize("bad_seed", [object(), 0.1])
def test_as_random_state_bad_seed(bad_seed: Any) -> None:
    """
    Parameters
    ----------
    bad_seed: Union[object, float]
        A seed that won't convert to RandomState

    Expects
    -------
    * Should return a random state object
    """
    with pytest.raises(ValueError):
        as_random_state(bad_seed)
