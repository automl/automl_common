from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, TypeVar, Union

from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble.builders.single_best import single_best
from automl_common.util.types import SupportsEqualty

import test.test_ensemble.test_builders.cases as cases

T = TypeVar("T", bound=SupportsEqualty)


@parametrize("predictions", [{}])
def test_predictions_empty_dict(predictions: Mapping[Hashable, np.ndarray]) -> None:
    """
    Parameters
    ----------
    predictions: Mapping[str, np.ndarray]
        An empty dictionary

    Expects
    -------
    * Should raise an error about an empty mapping
    """
    with pytest.raises(ValueError, match="`model_predictions` is empty"):
        single_best(
            model_predictions=predictions,
            targets=np.asarray([]),
            metric=lambda x, y: 42,
        )


@parametrize("best", ["loss", "score", "hell world", object(), []])
def test_bad_best_arg(best: str) -> None:
    """
    Parameters
    ----------
    best: str
        A bad `best` parameter str

    Expects
    -------
    * Should raise an error about a bad "best" parameter
    """
    with pytest.raises(ValueError, match="`best` must be"):
        single_best(
            model_predictions={"a": np.asarray([])},
            targets=np.asarray([]),
            metric=lambda x, y: 42,
            best=best,
        )


@parametrize("metric_args", [{"a": "apple", "b": "banana"}, None, {}])
def test_forward_metric_args(metric_args: Optional[Mapping[str, Any]]) -> None:
    """
    Parameters
    ----------
    metric_args: Mapping[Hashable, Any]
        Arguments to forward to a metric

    Expects
    -------
    * Should raise an error about a bad "best" parameter
    """
    mock_metric = MagicMock()
    mock_metric.return_value = 42

    model_predictions = {"a": np.asarray([1, 2, 3])}
    targets = np.asarray([1, 2, 3])
    single_best(
        model_predictions=model_predictions,
        targets=targets,
        metric=mock_metric,
        metric_args=metric_args,
    )

    # We convert None to an empty dict for consitency
    if metric_args is None:
        assert mock_metric.call_args.kwargs == {}
    else:
        assert mock_metric.call_args.kwargs == metric_args


@parametrize_with_cases(
    "model_predictions, targets, metric, best, expected", cases=cases, has_tag="single"
)
def test_single_best_is_chosen(
    model_predictions: Mapping[Hashable, np.ndarray],
    targets: np.ndarray,
    metric: Callable[..., T],
    best: Union[str, Callable[[Iterable[T]], T]],
    expected: Hashable,
) -> None:
    """
    Parameters
    ----------
    model_predictions: Mapping[Hashable, np.ndarray]
        The model predicitons

    target: np.ndarray
        The targets

    metric: Callable[..., T]
        The metric to perform between predictions and target

    best: Union[str, Callable[Iterable[T]], T]
        How to select the best from the scores generated

    expected: Hashable
        The expected id to be selected

    Expects
    -------
    * Should raise an error about a bad "best" parameter
    """
    assert expected in model_predictions

    chosen = single_best(
        model_predictions=model_predictions, targets=targets, metric=metric, best=best
    )

    assert chosen == expected
