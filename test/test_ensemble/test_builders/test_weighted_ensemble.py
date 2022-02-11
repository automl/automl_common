from typing import Callable, Mapping, Optional, Sequence, Tuple, TypeVar
from typing_extensions import Literal

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble.builders.weighted_ensemble import weighted_ensemble_caruana
from automl_common.util.types import Orderable

import test.test_ensemble.test_builders.cases as cases

OrderableT = TypeVar("OrderableT", bound=Orderable)


@parametrize("predictions", [{}])
def test_predictions_empty_dict(predictions: Mapping[str, np.ndarray]) -> None:
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
            select="min",
        )

    return  # pragma: no cover


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
            select="min",
        )

    return  # pragma: no cover


@parametrize_with_cases(
    "model_predictions, targets, metric, size, select," "expected_weights, expected_trajectory",
    cases=cases,
    has_tag="weighted",
)
def test_weighted_ensemble_is_chosen(
    model_predictions: Mapping[str, np.ndarray],
    targets: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], OrderableT],
    size: int,
    select: Literal["min", "max"],
    expected_weights: Mapping[str, float],
    expected_trajectory: Sequence[Tuple[str, float]],
    random_state: Optional[int] = None,
) -> None:
    """
    Parameters
    ----------
    model_predictions: Mapping[str, np.ndarray]
        The model predicitons

    target: np.ndarray
        The targets

    metric: Callable[[np.ndarray, np.ndarray], OrderableT]
        The metric to perform between predictions and target

    size: int
        The size of the weighted ensemble to produce

    select: "min" |  "max"
        How to select the best from the scores generated

    expected_weights: Mapping[str, float]
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
        select=select,
        random_state=random_state,
    )

    model_traj, perf_traj = zip(*expected_trajectory)
    expected_model_traj, expected_perf_traj = zip(*expected_trajectory)

    assert weights == expected_weights
    assert model_traj == expected_model_traj

    np.testing.assert_array_almost_equal(perf_traj, expected_perf_traj)
