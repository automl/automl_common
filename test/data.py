from typing import List, Optional, Tuple, Union
from typing_extensions import Literal

from dataclasses import dataclass

import numpy as np

from automl_common.util.random import as_random_state
from automl_common.data.math import normalize

DEFAULT_SEED = 42


@dataclass
class XYPack:
    x: np.ndarray
    y: np.ndarray


def xy(
    kind: Literal["classification", "regression"] = "regression",
    xdims: Tuple[int, ...] = (100, 3),
    ydims: Tuple[int, ...] = (100,),
    classes: Union[int, np.ndarray, List] = 2,
    targets: int = 1,
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

    targets: int = 1
        How many targets to use for regerssion, takes priority over ydim

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
            weights = normalize(weights)

        # Convert classes to be an np array of choices
        if isinstance(classes, int):
            assert classes >= 0
            classes = np.arange(0, classes + 1)

        elif isinstance(classes, (list, np.ndarray)):
            classes = np.array(classes)

        else:
            raise NotImplementedError()

        # Get x values
        x = rs.rand(*xdims)

        # If classes is multidimensional, we generate idxs of classes
        # and use those
        if classes.ndim > 1:
            idxs = rs.choice(len(classes), size=(len(x),), p=weights)
            y = classes[idxs]

        else:
            y = rs.choice(classes, size=ydims)

        return x, y

    elif kind == "regression":
        x = rs.rand(*xdims)

        if targets == 1:
            _ydims = ydims
        else:
            _ydims = (ydims[0], targets)

        y = rs.rand(*_ydims)

        return x, y

    else:
        raise NotImplementedError()


def arrhash(x: np.ndarray) -> int:
    """A collision prone hash, useful for mocks to produce same output on same arrays

    Parameters
    ----------
    x : np.ndarray
        The array to get a hash for

    Returns
    -------
    int
        The hash for the array
    """
    if x.ndim == 1:
        return int(sum(x[0:10]) if len(x) >= 10 else sum(x))
    else:
        return int(sum(x[0]))
