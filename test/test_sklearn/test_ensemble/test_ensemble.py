from typing import Callable, Tuple, Type

import numpy as np
from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.sklearn.ensemble import (
    Ensemble,
    SingleClassifierEnsemble,
    SingleRegressorEnsemble,
    WeightedClassifierEnsemble,
    WeightedRegressorEnsemble,
)

import test.test_sklearn.test_ensemble.cases as cases


@parametrize(
    "ensemble_type",
    [
        SingleRegressorEnsemble,
        SingleClassifierEnsemble,
        WeightedClassifierEnsemble,
        WeightedRegressorEnsemble,
    ],
)
def test_constructs_with_params(ensemble_type: Type[Ensemble]) -> None:
    """
    Parameters
    ----------
    ensemble_type : Type[Ensemble]
        The type to construct

    Expects
    -------
    * Should construct without problems with no parameters
    """
    ensemble_type()


@parametrize_with_cases("ensemble", cases=cases)
def test__fit_returns_ids(
    ensemble: Ensemble,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    make_xy : Callable[..., Tuple[np.ndarray, np.ndarray]]
        Factory to make x, y data

    Expects
    -------
    * Implementer should return a non-empty list of id's given valid training data
    """
    x, y = make_xy()
    ids = ensemble._fit(x, y)
    assert isinstance(ids, list) and len(ids) > 0


@parametrize_with_cases("ensemble", cases=cases)
def test__fit_does_not_modify_data(
    ensemble: Ensemble,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    make_xy : Callable[..., Tuple[np.ndarray, np.ndarray]]
        Factory to make x, y data

    Expects
    -------
    * Implementer should return a non-empty list of id's given valid training data
    """
    x, y = make_xy()
    x_, y_ = x.copy(), y.copy()
    ensemble._fit(x, y)

    np.testing.assert_equal(x, x_)
    np.testing.assert_equal(y, y_)


def test__predict_produces_results() -> None:
    ...


def test__predict_does_not_modify_data() -> None:
    ...


def test__fit_attributes_has_ids_() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_ids_when_fitted_is_valid() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_ids_when_not_fitted_raises_error() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, has_tag="empty_model_store")
def test_model_store_when_empty_raises_error() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_fit_sets_attributes() -> None:
    ...


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("empty_model_store"),
)
def test_fit_no_model_store_sets_fail_predict_shape_and_attr() -> None:
    ...


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted"),
)
def test_predict_when_not_fitted_raises_error() -> None:
    ...


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("empty_model_store"),
)
def test_predict_with_no_model_store_gives_correct_shape(
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]]
) -> None:
    ...


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("empty_model_store"),
)
def test_predict_gives_correct_shape(
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]]
) -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases)
def test_get_params_followed_by_set_params_sets_init_params(ensemble: Ensemble) -> None:
    """

    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to check

    Expects
    -------
    * Should be able to construct an ensemble of the same type as `ensemble` and use
        `get_params` on one with `set_params` on the other to align their params to be
        the same.
    """
    # Check one way
    ensemble_a = ensemble
    params_a = ensemble_a.get_params()

    ensemble_type = type(ensemble_a)
    ensemble_b = ensemble_type()

    ensemble_b.set_params(**params_a)
    params_b = ensemble_b.get_params()

    for p_b, val in params_b.items():
        assert params_a[p_b] == val

    # Check back
    ensemble_a.set_params(**params_b)
    params_a = ensemble_b.get_params()

    for p_a, val in params_a.items():
        assert params_b[p_a] == val


@parametrize_with_cases("ensemble", cases=cases, has_tag="empty_model_store")
def test_getitem_when_no_model_store_raises_error() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_getitem_when_not_fitted_raises_error() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_getitem_bad_model_id() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_getitem_can_load_all_items_in_iter() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_is_fitted_on_fitted() -> None:
    ...


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_is_fitted_on_non_fitted() -> None:
    ...
