"""
tags:
    {"weighted", "single"} - The type of case it is
"""

from typing import Any, Callable, TypeVar
from typing_extensions import Literal  # TODO, remove with Python 3.8

import numpy as np
from pytest_cases import case, parametrize

from automl_common.metrics import accuracy, rmse

T = TypeVar("T")


@case(tags=["single"])
@parametrize("metric, select", [(accuracy, "max"), (rmse, "min")])
def case_single_one_model_to_choose_with_predictions(
    metric: Callable[..., T],
    select: Literal["min", "max"],
) -> Any:
    """The case where the is only one model to choose from

    Should be the same model expected no matter what the metric or however the
    beset is selected from the metric.

    Parameters
    ----------
    metric: (preds, targets) -> T
        The metric that should be used

    select: "min" | "max"
        How to select the best from the list of scores
    """
    model_predictions = [("a", np.asarray([1, 1, 1]))]
    targets = np.asarray([0, 0, 0])
    expected = "a"

    return model_predictions, targets, metric, select, expected

    expected = "a"

    return model_predictions, targets, metric, select, expected


@case(tags=["single"])
@parametrize("n", [2, 5, 10])
@parametrize("metric, select", [(accuracy, "max"), (rmse, "min")])
def case_single_n_models(
    n: int,
    metric: Callable[..., T],
    select: Literal["min", "max"],
) -> Any:
    """The case where there is n models to choose from

    We randomly select the best model to ensure some fairness

    Parameters
    ----------
    n: int
        How many models to choose from

    metric: (preds, targets) -> T
        Metric to use to produce a score for a prediction

    select: "min" | "max"
        How to select the best from the list of scores
    """
    targets = np.asarray([0, 0, 0])
    model_predictions = [(str(i), np.asarray([1, 1, 0])) for i in range(n)]

    chosen = np.random.choice(list(range(n)))

    # The expected model is given the targets so they match exactly
    id, _ = model_predictions[chosen]
    model_predictions[chosen] = (id, targets)

    return model_predictions, targets, metric, select, id


@case(tags=["weighted"])
@parametrize("size", [1, 5, 10])
@parametrize("metric, select", [(accuracy, "max"), (rmse, "min")])
def case_weighted_one_model_to_choose(
    metric: Callable[..., T],
    select: Literal["min", "max"],
    size: int,
) -> Any:
    """The case where the is only one model to choose from

    Should be the same model expected no matter what the metric or however the
    best is chosen from the metric.

    Parameters
    ----------
    metric: (preds, targets) -> T
        The metric that should be used

    select: "min" | "max"
        How to select the best from the list of scores

    size: int
        How many members in the enseble
    """
    model_predictions = {"a": np.asarray([1, 1, 1])}
    targets = np.asarray([0, 0, 0])
    expected_weights = {"a": 1.0}

    res = metric(model_predictions["a"], targets)

    # Should expect that we get the same model and result n times
    expected_trajectory = [("a", res)] * size

    return (
        model_predictions,
        targets,
        metric,
        size,
        select,
        expected_weights,
        expected_trajectory,
    )


@case(tags=["weighted", "autosklearn"])
def case_autosklearn_weighted() -> Any:
    """The same test case as is used in autosklearn"""
    size = 10
    targets = np.full((100), 5.5)

    model_predictions = {str(i): np.full((100), i, dtype=np.float32) for i in range(1, 20)}

    for i, preds in enumerate(model_predictions.values(), start=1):
        preds[i * 5 : (i + 1) * 5] = 5.5 * i

    expected_weights = {
        "1": 0.1,
        "2": 0.2,
        "3": 0.2,
        "4": 0.1,
        "5": 0.1,
        "6": 0.1,
        "7": 0.1,
        "8": 0.1,
    }

    expected_trajectory = [
        ("3", 3.462296925452813),
        ("4", 2.679202306657711),
        ("5", 2.274862633215465),
        ("2", 2.065717187806695),
        ("6", 1.787456293171947),
        ("7", 1.698344782427879),
        ("2", 1.559451106993111),
        ("8", 1.531632605261457),
        ("1", 1.380195012164924),
        ("3", 1.355498063443839),
    ]
    metric = rmse
    select = "min"

    return (
        model_predictions,
        targets,
        metric,
        size,
        select,
        expected_weights,
        expected_trajectory,
    )
