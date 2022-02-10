from typing import Callable, List, Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np
from pytest_cases import fixture

from automl_common.util.random import as_random_state

DEFAULT_SEED = 42


def xy(
    kind: Literal["classification", "regression"] = "regression",
    xdims: Tuple[int, ...] = (100, 3),
    ydims: Tuple[int, ...] = (100,),
    classes: Union[int, np.ndarray, List] = [0, 1],
    weights: Optional[Union[np.ndarray, List[float]]] = None,
    random_state: Union[int, np.random.RandomState] = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate some arbitrary x,y data

    Parameters
    ----------
    kind : "classification" | "regression" = "regression"
        The kind of data to generate

    xdims: Tuple[int, ...] = (100, 3)
        Dimensions of x data

    ydims: Tuple[int, ...] = (100,)
        Dimensions of y data

    classes: int | np.ndarray | List = 2
        Classes to use if ``kind == "classification"``

    weights: Optional[np.ndarray | List[float]] = None
        Weights to apply to classes

    random_state: int | RandomState = DEFAULT_SEED
        Random state to use

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The generated data
    """
    assert xdims[0] == ydims[0]
    rs = as_random_state(random_state)

    if kind == "classification":
        # Normalize weights
        if weights:
            weights = np.array(weights) / np.sum(weights)

        # Convert classes to be an np array of choices
        if isinstance(classes, int):
            assert classes >= 0
            classes = np.arange(0, classes + 1)

        if isinstance(classes, list):
            classes = np.array(classes)

        # Get x values
        x = rs.rand(*xdims)

        # If classes is multidimensional, we generate idxs of classes
        # and use those
        if classes.ndim > 1:
            idxs = rs.choice(len(x), size=(ydims[0],), p=weights)
            y = classes[idxs]

        else:
            y = rs.choice(classes, size=ydims)

        return x, y

    elif kind == "regression":
        x = rs.rand(*xdims)
        y = rs.rand(*ydims)

        return x, y

    else:
        raise NotImplementedError()


@fixture(scope="function")
def make_xy() -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """Please see documentation of `def xy` in `test/data.py`"""
    return xy
