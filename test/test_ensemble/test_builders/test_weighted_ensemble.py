from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble.builders.weighted_ensemble import weighted_ensemble_caruana
from automl_common.util.types import SupportsEqualty

import test.test_ensemble.test_builders.cases as cases

T = TypeVar("T", bound=SupportsEqualty)
ID = TypeVar("ID", bound=Hashable)


@parametrize("predictions", [{}])
def test_predictions_empty_dict(predictions: Mapping[ID, np.ndarray]) -> None:
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
        weighted_ensemble_caruana(
            model_predictions=predictions,
            size=1,
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
        weighted_ensemble_caruana(
            model_predictions={"a": np.asarray([])},
            targets=np.asarray([]),
            size=1,
            metric=lambda x, y: 42,
            best=best,
        )


@parametrize("size", [-5, -1, 0])
def test_bad_size(size: int) -> None:
    """
    Parameters
    ----------
    size: int
        A bad size argument

    Expects
    -------
    * Should not be able to fit an ensemble of size 0
    """
    with pytest.raises(ValueError, match="`size` must be"):
        weighted_ensemble_caruana(
            model_predictions={"a": np.asarray([])},
            targets=np.asarray([]),
            size=size,
            metric=lambda x, y: 42,
        )


@parametrize("metric_args", [{"a": "apple", "b": "banana"}, None, {}])
def test_forward_metric_args(metric_args: Optional[Mapping[str, Any]]) -> None:
    """
    Parameters
    ----------
    metric_args: Mapping[str, Any]
        Arguments to forward to a metric

    Expects
    -------
    * Should raise an error about a bad "best" parameter
    """
    mock_metric = MagicMock()
    mock_metric.return_value = 42

    model_predictions = {"a": np.asarray([1, 2, 3])}
    targets = np.asarray([1, 2, 3])
    weighted_ensemble_caruana(
        model_predictions=model_predictions,
        targets=targets,
        size=1,
        metric=mock_metric,
        metric_args=metric_args,
    )

    # We convert None to an empty dict for consitency
    if metric_args is None:
        assert mock_metric.call_args.kwargs == {}
    else:
        assert mock_metric.call_args.kwargs == metric_args


@parametrize_with_cases(
    "model_predictions, targets, metric, size, best, "
    "expected_weights, expected_trajectory",
    cases=cases,
    has_tag="weighted",
)
def test_weighted_ensemble_is_chosen(
    model_predictions: Mapping[ID, np.ndarray],
    targets: np.ndarray,
    metric: Callable[..., T],
    size: int,
    best: Union[str, Callable[[Iterable[T]], T]],
    expected_weights: Mapping[ID, float],
    expected_trajectory: Sequence[Tuple[ID, float]],
    random_state: Optional[int] = None,
) -> None:
    """
    Parameters
    ----------
    model_predictions: Mapping[ID, np.ndarray]
        The model predicitons

    target: np.ndarray
        The targets

    metric: Callable[..., T]
        The metric to perform between predictions and target

    size: int
        The size of the weighted ensemble to produce

    best: Union[str, Callable[Iterable[T]], T]
        How to select the best from the scores generated

    expected_weights: Mapping[ID, float]
        The expected id to be selected

    expected_trajectory: Iterable[T]
        The expected trajectory it would take

    Expects
    -------
    * Should choose the expected ensemble with their weights
    """
    assert set(expected_weights).issubset(set(model_predictions))
    assert {id for id, _ in expected_trajectory}.issubset(set(model_predictions))

    weights, trajectory = weighted_ensemble_caruana(
        model_predictions=model_predictions,
        targets=targets,
        size=size,
        metric=metric,
        best=best,
        random_state=random_state,
    )

    model_traj, perf_traj = zip(*expected_trajectory)
    expected_model_traj, expected_perf_traj = zip(*expected_trajectory)

    assert weights == expected_weights
    assert model_traj == expected_model_traj

    np.testing.assert_array_almost_equal(perf_traj, expected_perf_traj)
