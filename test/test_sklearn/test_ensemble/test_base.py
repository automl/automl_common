from typing import Callable, List, Tuple

import numpy as np
from sklearn.exceptions import NotFittedError

from automl_common.sklearn.ensemble import Ensemble

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases
from test.test_sklearn.test_ensemble.cases import cases

DataFactory = Callable[..., Tuple[np.ndarray, np.ndarray]]


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_ids_when_fitted_gives_no_error(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    Expects
    -------
    * Should produce a list of ids

    Note
    ----
    This will be an empty list when created with `model_store` is None but it is
    technically fitted and should not give an error.
    """
    ensemble.ids


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_ids_when_not_fitted_raises_error(ensemble: Ensemble) -> None:
    """

    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    Expects
    -------
    * Should not be able to access ids if not fitted
    """
    with pytest.raises(NotFittedError):
        ensemble.ids

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_fit_sets_attributes(ensemble: Ensemble, make_xy: DataFactory) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    make_xy : DataFactory
        Factory to make x,y data
    """
    x, y = make_xy(kind="classification")  # Regression algorithms should be able to fit to
    ensemble.fit(x, y)

    assert all(hasattr(ensemble, attr) for attr in ensemble._fit_attributes())


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_fit_does_not_modify_data(
    ensemble: Ensemble,
    make_xy: DataFactory,
) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    make_xy: DataFactory
        Make x, y data

    Expects
    -------
    * Implementer should return a non-empty list of id's given valid training data
    """
    x, y = make_xy("classification")
    x_, y_ = x.copy(), y.copy()
    ensemble.fit(x, y)

    np.testing.assert_equal(x, x_)
    np.testing.assert_equal(y, y_)


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_predict_does_not_modify_data(ensemble: Ensemble, make_xy: DataFactory) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    make_xy : DataFactory
        Factory to make x,y data

    Expects
    -------
    * The ensemble should not modify data while predicting
    """
    x, _ = make_xy()
    x_copy = x.copy()

    ensemble.predict(x_copy)

    np.testing.assert_equal(x_copy, x)


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted"),
)
def test_predict_when_not_fitted_raises_error(
    ensemble: Ensemble,
) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        An unfitted ensemble

    Expects
    -------
    * Should raise attribute erorr, not fitted
    """
    with pytest.raises(NotFittedError):
        ensemble.predict(np.array([1, 2, 3, 4]))

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_getitem_when_not_fitted_raises_error(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        An unfitted ensemble

    Expects
    -------
    * Should raise  NotFittedError, not fitted
    """
    with pytest.raises(NotFittedError):
        ensemble["badkey"]

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_getitem_bad_model_id(ensemble: Ensemble) -> None:
    assert "badkey" not in ensemble
    with pytest.raises(KeyError):
        ensemble["badkey"]

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_models_in_iter_can_be_gotten(ensemble: Ensemble) -> None:
    """

    Parameters
    ----------
    ensemble : Ensemble
        A fitted ensemble

    Expects
    -------
    * Should be able to unpack in a dict like fashion
    * All names in iter should be in this unpack
    * All names in iter should be able to be used with `ensemble[name]`
    """
    model_dict = {**ensemble}

    assert all(name in model_dict for name in iter(ensemble))
    assert all(ensemble[name] is not None for name in iter(ensemble))


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_fit_twice_produces_same_attributes(
    ensemble: Ensemble,
    make_xy: DataFactory,
) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        An unfitted ensemble

    make_xy: DataFactory
        Factory for making x, y data

    Expects
    -------
    * The ensemble fitted twice to the same data should produce the same attributes
    """
    x, y = make_xy(kind="classification")

    ensemble.fit(x, y)
    first_fit = {attr: getattr(ensemble, attr) for attr in ensemble._fit_attributes()}

    ensemble.fit(x, y)
    second_fit = {attr: getattr(ensemble, attr) for attr in ensemble._fit_attributes()}

    assert set(first_fit) == set(second_fit)
    attrs = list(first_fit)

    # Can't compare random state attr
    if "random_state_" in attrs:
        attrs.remove("random_state_")

    for attr in attrs:
        first, second = first_fit[attr], second_fit[attr]
        if isinstance(first, (np.ndarray, List)):
            np.testing.assert_array_equal(first, second)
        else:
            assert first_fit[attr] == second_fit[attr]
