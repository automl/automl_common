from typing import Any, Callable, Mapping, Optional

import numpy as np

# https://github.com/python/mypy/issues/5876#issuecomment-706396563

Metric = Callable[..., float]
# (np.ndarray, np.ndarray, kwargs) -> float


def single_best(
    model_predictions: Mapping[str, np.ndarray],
    y: np.ndarray,
    metric: Metric,
    metric_args: Optional[Mapping[str, Any]] = None,
) -> str:
    """Get an Ensemble consisting of the single best model

    Parameters
    ----------
    model_predictions: Mapping[str, np.ndarray]
        A Mapping from model ids to their predictions

    y: np.ndarray
        The targets

    metric: (np.ndarray, np.ndarray, np.ndarray=None) -> float
        A metric to asses predictions on. Must take in the following order

        pred: np.ndarray
            The predictions created by a model

        target: np.ndarray
            The corresponding targets

        sample_weights: Optional[np.ndarray] = None
            The sample weights to pass to the metric

        returns: float
            The score of the model

    metric_args: Dict[str, Any]
        Arguments to forward to the metric

    Returns
    -------
    str

    """
    metric_args = {} if metric_args is None else metric_args
    scores = {
        id: metric(prediction, y, **metric_args)
        for id, prediction in model_predictions.items()
    }

    best_id = max(scores, key=lambda id: scores[id])
    return best_id
