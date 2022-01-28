from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import logging
from collections import Counter

import numpy as np

from automl_common.util import as_random_state
from automl_common.util.types import SupportsEqualty

logger = logging.getLogger(__name__)

# Values return by metric require that we can perform equality checks on them
T = TypeVar("T", bound=SupportsEqualty)
ID = TypeVar("ID", bound=Hashable)

Trajectory = List[Tuple[ID, T]]


def weighted_ensemble_caruana(
    model_predictions: Mapping[ID, np.ndarray],
    targets: np.ndarray,
    size: int,
    metric: Callable[..., T],  # TODO Python 3.10, update params with PEP 612
    metric_args: Optional[Mapping[str, Any]] = None,
    best: Union[str, Callable[[Iterable[T]], T]] = "max",
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[Dict[ID, float], Trajectory]:
    """Calculate a weighted ensemble of `n` models

    Parameters
    ----------
    model_predictions: Mapping[Hashable, np.ndarray]
        The model predictions to use, mapping from id to their predictions

    targets: np.ndarray
        The targets

    size: int
        The size of the ensemble to create

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
    (Dict[str, float], List[T])
        A dictionary mapping from id's to values genrated from adding a model at each
        time step.
    """
    if not size > 0:
        raise ValueError("`size` must be positive")

    if len(model_predictions) == 0:
        raise ValueError("`model_predictions` is empty")

    if not callable(best) and best not in ("max", "min"):
        raise ValueError("`best` must be either 'max' or 'min' or a Callable")

    # Get the `best` function
    if callable(best):  # isinstance(best, Callable) does not work
        get_best_val = best
    elif best == "max":
        get_best_val = max  # type: ignore
    elif best == "min":
        get_best_val = min  # type: ignore
    else:
        raise NotImplementedError

    rand = as_random_state(random_state)
    kwargs = metric_args if metric_args is not None else {}

    predictions = list(model_predictions.values())

    dtype = predictions[0].dtype
    if np.issubdtype(dtype, np.integer):
        dtype = np.float64
        logger.warning(
            f"Predictions were {predictions[0].dtype}, converting to {dtype} to"
            " allow for weighted ensemble procedure"
        )

    # Current sum of predictions in the ensemble
    current = np.zeros_like(predictions[0], dtype=dtype)

    # Buffer where new models predictions are added to current to try them
    buffer = np.empty_like(predictions[0], dtype=dtype)

    ensemble: List[ID] = []
    trajectory: Trajectory = []

    def value_if_added(_pred: np.ndarray) -> T:
        # Get the value if the model was added to the current set of predicitons
        np.add(current, _pred, out=buffer)
        np.multiply(buffer, (1.0 / float(len(ensemble) + 1)), out=buffer)
        return metric(buffer, targets, **kwargs)

    for i in range(size):
        # Get the value if added for each model
        scores = {id: value_if_added(pred) for id, pred in model_predictions.items()}

        # Get the choices that produce the best value
        best_val = get_best_val(iter(scores.values()))
        best_choices = [id for id, score in scores.items() if score == best_val]

        # Select one
        chosen_idx = rand.choice(len(best_choices))
        chosen = best_choices[chosen_idx]

        # Add the predictions of the chosen model
        np.add(current, model_predictions[chosen], out=current)

        # Record it's addition and the over all trajectory of loss
        ensemble.append(chosen)
        trajectory.append((chosen, best_val))

        # In the case of only one model, have calculated it's loss
        # and it's the only available model to add to the ensemble
        if len(model_predictions) == 1:
            trajectory *= size
            ensemble *= size
            break

    # Calculate weights
    weighted_ensemble = {id: count / size for id, count in Counter(ensemble).items()}

    return weighted_ensemble, trajectory
