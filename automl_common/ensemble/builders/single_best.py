from typing import Any, Mapping, Optional

import numpy as np

from automl_common.metrics import MetricProtocol


def single_best(
    model_predictions: Mapping[str, np.ndarray],
    y: np.ndarray,
    metric: MetricProtocol,
    metric_args: Optional[Mapping[str, Any]] = None,
) -> str:
    """Get an Ensemble consisting of the single best model

    Parameters
    ----------
    model_predictions: Mapping[str, np.ndarray]
        A Mapping from model ids to their predictions

    y: np.ndarray
        The targets

    metric: MetricProtocol
        A metric to asses predictions on. Must take in the following order

        pred: np.ndarray
            The predictions created by a model

        target: np.ndarray
            The corresponding targets

        **kwargs
            Any other values to forward to the metric

        returns: float
            The score of the model

    metric_args: Optional[Mapping[str, Any]] = None
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
