from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union
from typing_extensions import Literal

import logging
from collections import Counter

import numpy as np

from automl_common.data.convert import probabilities_to_classes
from automl_common.util.random import as_random_state
from automl_common.util.types import Orderable

logger = logging.getLogger(__name__)

# Values return by metric require that we can perform equality checks on them
OrderableT = TypeVar("OrderableT", bound=Orderable)
ID = TypeVar("ID")


def weighted_ensemble_caruana(
    model_predictions: Mapping[ID, np.ndarray],
    targets: np.ndarray,
    size: int,
    metric: Callable[[np.ndarray, np.ndarray], OrderableT],
    select: Literal["min", "max"],
    is_probabilities: bool = False,
    classes: Optional[Union[List, np.ndarray]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[Dict[ID, float], List[Tuple[ID, OrderableT]]]:
    """Calculate a weighted ensemble of `n` models

    Note
    ----
    If passing probabilities for for ``model_predictions``, must
    specify ``is_probabilities = True`` and provide ``classes``.

    Parameters
    ----------
    model_predictions: Mapping[Hashable, np.ndarray]
        The model predictions to use, mapping from id to their predictions

    is_probabilities: bool = False
        Whether the predictions are probabilities

    targets: np.ndarray
        The targets

    size: int
        The size of the ensemble to create

    metric: (pred: np.ndarray, target: np.ndarray) -> OrderableT
        The metric to use in calculating which models to add to the ensemble.
        Should retunr something orderable

    select: (Dict[ID, OrderableT]) -> ID | List[ID]
        Selects a models from the list based on the values of the metric on their
        predictions. Can return a single ID or a list of them, in which case a
        random selection will be made.

    classes: Optional[np.ndarray | List] = None
        The classes to use if ``is_probabilities`` for ``model_predictions``. For
        now we assume to style of sklearn for specifying clases and probabilties
        for binary, multiclass and multi-label targets.

    random_state: Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties

    Returns
    -------
    (Dict[ID, float], List[Tuple[ID, OrderableT]])
        A dictionary mapping from id's to values genrated from adding a model at each
        time step.
    """
    if not size > 0:
        raise ValueError("`size` must be positive")

    if len(model_predictions) == 0:
        raise ValueError("`model_predictions` is empty")

    if is_probabilities is True and classes is None:
        raise ValueError("Must provide `classes` if using probabilities")

    if is_probabilities is False and classes is not None:
        raise ValueError("`classes` should not be provided if not using probabilities")

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

    ensemble: List[ID] = []
    trajectory: List[Tuple[ID, OrderableT]] = []

    def value_if_added(_pred: np.ndarray) -> OrderableT:
        # Get the value if the model was added to the current set of predicitons
        np.add(current, _pred, out=buffer)
        np.multiply(buffer, (1.0 / float(len(ensemble) + 1)), out=buffer)

        # We need to convert the probabilities to classes before passing to metric
        if is_probabilities:
            assert classes is not None
            predictions = probabilities_to_classes(buffer, classes)
            return metric(predictions, targets)
        else:
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
        choice = rand.choice(np.asarray(list(choices)))

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
