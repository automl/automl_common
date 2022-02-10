from typing import Callable, Iterable, Optional, Tuple, Union
from typing_extensions import Literal  # TOOD: remove python 3.8

import numpy as np

from automl_common.util import as_random_state
from automl_common.util.types import Orderable


def single_best(
    model_predictions: Iterable[Tuple[str, np.ndarray]],
    targets: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], Orderable],
    select: Literal["min", "max"],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> str:
    """Get an Ensemble consisting of the single best model

    Parameters
    ----------
    model_predictions: Iterable[Tuple[str, np.ndarray]]
        A interable of model ids and predictions

    targets: np.ndarray
        The targets

    metric: (pred: np.ndarray, target: np.ndarray) -> Orderable
        The metric to use in calculating which models to add to the ensemble.
        Should return something that can be ordered

    select: (Dict[str, T]) -> str | List[str]
        Selects a models from the list based on the values of the metric on their
        predictions. Can return a single str or a list of them, in which case a
        random selection will be made.

    random_state: Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties

    Returns
    -------
    str
        The id of the chosen model
    """
    scores = {id: metric(prediction, targets) for id, prediction in model_predictions}

    if len(scores) == 0:
        raise ValueError("`model_predictions` was empty")

    rand = as_random_state(random_state)

    best_val = min(scores.values()) if select == "min" else max(scores.values())
    choices = [id for id, score in scores.items() if score == best_val]
    choice = rand.choice(np.array(list(choices)))

    return choice
