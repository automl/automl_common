from typing import Callable, Tuple, Type

from unittest.mock import patch

import numpy as np
import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.sklearn.ensemble import (
    ClassifierEnsemble,
    Ensemble,
    SingleClassifierEnsemble,
    SingleRegressorEnsemble,
    WeightedClassifierEnsemble,
    WeightedRegressorEnsemble,
)

import test.test_sklearn.test_ensemble.cases as cases

from sklearn.exceptions import NotFittedError

DataFactory = Callable[..., Tuple[np.ndarray, np.ndarray]]
fit_err = r"Please call `fit` first"


ensemble_types = [
    SingleRegressorEnsemble,
    SingleClassifierEnsemble,
    WeightedClassifierEnsemble,
    WeightedRegressorEnsemble,
]


@parametrize("ensemble_type", ensemble_types)
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


@parametrize("ensemble_type", ensemble_types)
def test_fit_with_no_params_causes_error(ensemble_type: Type[Ensemble]) -> None:
    """
    Parameters
    ----------
    ensemble_type : Type[Ensemble]
        The type to construct

    Expects
    -------
    * Should construct without problems with no parameters
    """
    ensemble = ensemble_type()
    with pytest.raises(RuntimeError, match="Can't fit without model store"):
        ensemble.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))

    return  # pragma: no cover


@parametrize("ensemble_type", ensemble_types)
def test__model_store_raises_when_not_fitted(ensemble_type: Type[Ensemble]) -> None:
    """
    Parameters
    ----------
    ensemble_type : Type[Ensemble]
        An ensemble type to construct

    Expects
    -------
    * The _model_store should give an attribute error when not fitted
    """
    ensemble = ensemble_type()
    with pytest.raises(AttributeError):
        ensemble._model_store

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, filter=ft.has_tag("fitted"))
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
    x, y = make_xy(kind="classification")  # Regression algorithms should be able to fit
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
    x, y = make_xy()
    x_, y_ = x.copy(), y.copy()
    ensemble._fit(x, y)

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


@parametrize_with_cases("ensemble", cases=cases, has_tag="fitted")
def test_is_fitted_on_fitted(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        The ensemble to test

    Expects
    -------
    * All ensembles that were fitted should advertise fitted
    """
    assert ensemble.__sklearn_is_fitted__()


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_is_fitted_on_non_fitted(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble

    Expects
    -------
    * All models that were not fitted should advertise not fitted
    """
    assert not ensemble.__sklearn_is_fitted__()


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("classifier"),
)
def test_predict_proba_raises_when_not_fitted(ensemble: ClassifierEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : ClassifierEnsemble
        An unfitted ClassifierEnsemble

    Expects
    -------
    * The unfitted ensemble should not be able to predict_proba
    """
    with pytest.raises(NotFittedError):
        ensemble.predict_proba(np.array([1, 2, 3, 4]))

    return  # pragma: no cover


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("classifier"),
)
def test_predict_proba_does_not_modify_data(
    ensemble: ClassifierEnsemble,
    make_xy: DataFactory,
) -> None:
    """
    Parameters
    ----------
    ensemble : ClassifierEnsemble
        An unfitted ClassifierEnsemble

    make_xy: DataFactory
        Factory for making x, y data

    Expects
    -------
    * The ensemble predict_proba should not modify data
    """
    x, y = make_xy(kind="classification")
    x_ = x.copy()

    ensemble.fit(x, y)
    ensemble.predict_proba(x)

    np.testing.assert_equal(x, x_)


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_fit_twice_produces_same_attributes(
    ensemble: ClassifierEnsemble,
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

    for name, model in ensemble.model_store.items():
        print(model.load())
    ensemble.fit(x, y)
    first_fit = {attr: getattr(ensemble, attr, None) for attr in ensemble._fit_attributes()}

    ensemble.fit(x, y)
    second_fit = {attr: getattr(ensemble, attr, None) for attr in ensemble._fit_attributes()}

    assert set(first_fit) == set(second_fit)
    for attr in first_fit:
        if attr == "random_state_":
            continue

        assert first_fit[attr] == second_fit[attr]


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("classifier") & ft.has_tag("fitted"),
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_proba_raises_with_jagged_predictions(ensemble: ClassifierEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        A fitted classifier ensemble

    Expects
    -------
    * A runtime error should be raise as the probability predictions don't have the
      correct size
    """
    jagged = np.array([[1, 2, 3], [1, 2]])
    with patch.object(ensemble, "_predict_proba", return_value=jagged):

        with pytest.raises(RuntimeError, match="Probability predictions were jagged"):
            x = np.array([[1, 2, 3], [1, 2, 3]])
            ensemble.predict_proba(x)

    return  # pragma: no cover
