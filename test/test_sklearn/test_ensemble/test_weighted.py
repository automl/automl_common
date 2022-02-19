from typing import List, Tuple

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

import numpy as np
from sklearn.exceptions import NotFittedError

from automl_common.sklearn.ensemble import WeightedEnsemble

from test.test_sklearn.test_ensemble.cases import cases


@parametrize_with_cases("ensemble", cases=cases, has_tag=["fitted", "weighted"])
def test_trajectory(ensemble: WeightedEnsemble) -> None:
    """
    Note
    ----
    This test is not fully accurate as the DummyClassifier and DummyRegressor
    will always return the same results. We do however test this properly
    in `test_builders/cases.py::case_autosklearn_weighted()`.

    Parameters
    ----------
    ensemble: WeightedEnsemble
        A fitted weighted ensemble

    Expects
    -------
    * Should be able to get tracjectory_ without issue
    * Trajectory length should be same as it's size
    * The trajectory should be sorted either min -> max or max -> min
    """
    traj: List[Tuple[str, float]] = ensemble.trajectory  # type: ignore

    assert len(traj) == ensemble.size

    scores = [score for _, score in traj]

    success = False
    try:
        np.testing.assert_allclose(scores, sorted(scores))
        success = True
    except AssertionError:
        np.testing.assert_allclose(scores, sorted(scores, reverse=True))
        success = True

    assert success


@parametrize_with_cases("ensemble", cases=cases, has_tag=["fitted", "weighted"])
def test_weights(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A fitted ensemble

    Expects
    -------
    * The weights should add up to 1
    * Each model in weight should exist in the ensemble
    """
    assert sum(ensemble.weights.values()) == pytest.approx(1)
    assert all(model_name in ensemble for model_name in ensemble.weights)


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ~ft.has_tag("fitted"),
)
def test_properties_when_not_fitted_raise_error(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble: WeightedEnsemble
        The not fitted weighted ensemble

    Expects
    -------
    * Should raise a NotFittedError when accessing it's properties
    """
    properties = ["trajectory", "weights"]
    for property in properties:
        with pytest.raises(NotFittedError):
            getattr(ensemble, property)

    return  # pragma: no cover


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ~ft.has_tag("fitted"),
)
def test_weights_when_not_fitted(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A non-fitted ensemble

    Expects
    -------
    * Accessing weights should raise NotFittedError when ensemble is not fitted
    """
    with pytest.raises(NotFittedError):
        ensemble.weights

    return  # pragma: no cover
