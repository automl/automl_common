from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union
from typing_extensions import Literal

import logging
from collections import Counter

import numpy as np

from automl_common.util.random import as_random_state
from automl_common.util.types import Orderable

logger = logging.getLogger(__name__)

# Values return by metric require that we can perform equality checks on them
OrderableT = TypeVar("OrderableT", bound=Orderable)


def weighted_ensemble_caruana(
    model_predictions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    size: int,
    metric: Callable[[np.ndarray, np.ndarray], OrderableT],
    select: Literal["min", "max"],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, OrderableT]]]:
    """Calculate a weighted ensemble of `n` models

    Parameters
    ----------
    model_predictions: Mapping[Hashable, np.ndarray]
        The model predictions to use, mapping from id to their predictions

    targets: np.ndarray
        The targets

    size: int
        The size of the ensemble to create

    metric: (pred: np.ndarray, target: np.ndarray) -> OrderableT
        The metric to use in calculating which models to add to the ensemble.
        Should retunr something orderable

    select: (Dict[str, OrderableT]) -> str | List[str]
        Selects a models from the list based on the values of the metric on their
        predictions. Can return a single str or a list of them, in which case a
        random selection will be made.

    random_state: Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties

    Returns
    -------
    (Dict[str, float], List[Tuple[str, OrderableT]])
        A dictionary mapping from id's to values genrated from adding a model at each
        time step.
    """
    if not size > 0:
        raise ValueError("`size` must be positive")

    if len(model_predictions) == 0:
        raise ValueError("`model_predictions` is empty")

    rand = as_random_state(random_state)

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

    ensemble: List[str] = []
    trajectory: List[Tuple[str, OrderableT]] = []

    def value_if_added(_pred: np.ndarray) -> OrderableT:
        # Get the value if the model was added to the current set of predicitons
        np.add(current, _pred, out=buffer)
        np.multiply(buffer, (1.0 / float(len(ensemble) + 1)), out=buffer)
        return metric(buffer, targets)

    for i in range(size):
        # Get the value if added for each model
        scores = {id: value_if_added(pred) for id, pred in model_predictions.items()}

        # Get the choices that produce the best value
        if select == "min":
            best_val = min(scores.values())
        elif select == "max":
            best_val = max(scores.values())
        else:
            raise NotImplementedError()

        choices = [id for id, score in scores.items() if score == best_val]
        choice = rand.choice(np.array(list(choices)))

        # Add the predictions of the chosen model
        np.add(current, model_predictions[choice], out=current)

        # Record it's addition and the over all trajectory of loss
        ensemble.append(choice)
        trajectory.append((choice, scores[choice]))

        # In the case of only one model, have calculated it's loss
        # and it's the only available model to add to the ensemble
        if len(model_predictions) == 1:
            trajectory *= size
            ensemble *= size
            break

    # Calculate weights
    weighted_ensemble = {id: count / size for id, count in Counter(ensemble).items()}

    return weighted_ensemble, trajectory
