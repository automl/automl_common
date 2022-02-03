from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from automl_common.util import as_random_state
from automl_common.util.types import SupportsEqualty

T = TypeVar("T", bound=SupportsEqualty)  # Metric result type
ID = TypeVar("ID", bound=Hashable)


def single_best(
    model_predictions: Iterable[Tuple[ID, np.ndarray]],
    targets: np.ndarray,
    metric: Callable[..., T],  # TODO Python 3.10, update params with PEP 612
    metric_args: Optional[Mapping[str, Any]] = None,
    best: Union[str, Callable[[Iterable[T]], T]] = "max",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> ID:
    """Get an Ensemble consisting of the single best model

    Parameters
    ----------
    model_predictions: Iterable[Tuple[ID, np.ndarray]]
        A interable of model ids and predictions

    targets: np.ndarray
        The targets

    metric: (pred: np.ndarray, target: np.ndarray, ...) -> T
        The metric to use in calculating which models to add to the ensemble. Must
        return a `T` that can be compared with `==`. This could be useful for
        using multiple metric and return a tuple such as `(x, y, z)`.

    metric_args: Optional[Mapping[str, Any]] = None
        Arguments to forward to the metric

    best: "min" | "max" | (Iterable[T]) -> T = "max"
        Select a model member at each stage according to the "min" or "max" of the score
        when adding the model.

        Optionally, you can pass your own `best` function that accepts the output of
        `metric` and returns a `T` which supports equality `==`. This could be useful
        for using multiple metric and return a tuple such as `(x, y, z)`.

    random_state: Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties

    Returns
    -------
    ID
        The id of the chosen model
    """
    if not callable(best) and best not in ("max", "min"):
        raise ValueError("`best` must be either 'max' or 'min' or a Callable")

    # Get the `best` function
    if callable(best):
        get_best_val = best
    elif best == "max":
        get_best_val = max  # type: ignore
    elif best == "min":
        get_best_val = min  # type: ignore
    else:
        raise NotImplementedError

    metric_args = {} if metric_args is None else metric_args

    scores = {
        id: metric(prediction, targets, **metric_args)
        for id, prediction in model_predictions
    }

    if len(scores) == 0:
        raise ValueError("`model_predictions` was empty")

    rand = as_random_state(random_state)

    best_val = get_best_val(iter(scores.values()))
    best_choices = [id for id, score in scores.items() if score == best_val]

    chosen_idx = rand.choice(len(best_choices))

    return best_choices[chosen_idx]
