from typing import Callable, List, Tuple, TypeVar
from typing_extensions import Literal  # TODO python3.8

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble.builders.single_best import single_best
from automl_common.util.types import Orderable

import test.test_ensemble.test_builders.cases as cases

OrderableT = TypeVar("OrderableT", bound=Orderable)


@parametrize("predictions", [[]])
def test_predictions_empty_dict(predictions: List[Tuple[str, np.ndarray]]) -> None:
    """
    Parameters
    ----------
    predictions: List
        An empty List

    Expects
    -------
    * Should raise an error about an empty mapping
    """
    with pytest.raises(ValueError, match="`model_predictions` was empty"):
        single_best(
            model_predictions=predictions,
            targets=np.asarray([]),
            metric=lambda x, y: 42,
            select="min",
        )

    return  # pragma: no cover


@parametrize_with_cases(
    "model_predictions, targets, metric, select, expected",
    cases=cases,
    has_tag="single",
)
def test_single_best_is_chosen(
    model_predictions: List[Tuple[str, np.ndarray]],
    targets: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], OrderableT],
    select: Literal["min", "max"],
    expected: str,
) -> None:
    """
    Parameters
    ----------
    model_predictions: List[Tuple[str, np.ndarray]]
        The model predicitons

    target: np.ndarray
        The targets

    metric: Callable[[np.ndarray, np.ndarray], OrderableT]
        The metric to perform between predictions and target

    select: "min" | "max"
        How to select the best from the scores generated

    expected: str
        The expected id to be selected

    Expects
    -------
    * Should chose the expected model
    """
    ids, _ = zip(*model_predictions)
    assert expected in ids

    chosen = single_best(
        model_predictions=model_predictions,
        targets=targets,
        metric=metric,
        select=select,
    )

    assert chosen == expected
