from typing import Callable, Hashable, Iterable, List, Tuple, TypeVar, Union

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble.builders.single_best import single_best
from automl_common.util.types import SupportsEqualty

import test.test_ensemble.test_builders.cases as cases

T = TypeVar("T", bound=SupportsEqualty)


@parametrize("predictions", [[]])
def test_predictions_empty_dict(predictions: List[Tuple[Hashable, np.ndarray]]) -> None:
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
            model_predictions=[("a", np.asarray([]))],
            targets=np.asarray([]),
            metric=lambda x, y: 42,
            best=best,
        )


@parametrize_with_cases(
    "model_predictions, targets, metric, best, expected",
    cases=cases,
    has_tag="single",
)
def test_single_best_is_chosen(
    model_predictions: List[Tuple[Hashable, np.ndarray]],
    targets: np.ndarray,
    metric: Callable[..., T],
    best: Union[str, Callable[[Iterable[T]], T]],
    expected: Hashable,
) -> None:
    """
    Parameters
    ----------
    model_predictions: List[Tuple[Hashable, np.ndarray]]
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
    ids, _ = zip(*model_predictions)
    assert expected in ids

    chosen = single_best(
        model_predictions=model_predictions,
        targets=targets,
        metric=metric,
        best=best,
    )

    assert chosen == expected
