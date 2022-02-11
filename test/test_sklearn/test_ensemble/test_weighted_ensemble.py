import pytest
from pytest import approx
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.sklearn.ensemble import WeightedEnsemble

import test.test_sklearn.test_ensemble.cases as cases


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("fitted"),
)
def test_trajectory_is_size_of_ensemble(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A weighted ensemble that's been fitted

    Expects
    -------
    * There should be one entry in trajectory per model added
    """
    assert len(ensemble.trajectory) == ensemble.size


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("fitted"),
)
def test_trajectory_is_sorted(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A weighted ensemble that's been fitted

    Expects
    -------
    * The entries in the trajectory should be sorted
    """
    scores = [x for _, x in ensemble.trajectory]
    assert scores == sorted(scores)


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("weighted"),
)
def test_trajectory_raises_when_not_fitted(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A singele ensemble that has not been fitted

    Expects
    -------
    * Should give an attribute error if accessing this on a non fitted ensemble
    """
    with pytest.raises(AttributeError):
        ensemble.trajectory


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("fitted"),
)
def test_sum_of_weights_is_one(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A weighted ensemble that's been fitted

    Expects
    -------
    * The weights should be normalized so they add to 1
    """
    assert sum(ensemble.weights.values()) == approx(1, abs=1e-7)


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("weighted"),
)
def test_weights_raises_when_not_fitted(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : WeightedEnsemble
        A weighted ensemble that's not been fitted

    Expects
    -------
    * The entries in the trajectory should be sorted
    """
    with pytest.raises(AttributeError):
        ensemble.weights

    return  # pragma: no cover
