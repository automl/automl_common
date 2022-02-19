from typing import Callable, List, Tuple, Type, Union

from pathlib import Path

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

import numpy as np
from sklearn.exceptions import NotFittedError

from automl_common.backend.stores.model_store import ModelStore
from automl_common.sklearn.ensemble import (
    ClassifierEnsemble,
    SingleClassifierEnsemble,
    WeightedClassifierEnsemble,
)

import test.test_sklearn.test_ensemble.test_classification.cases as cases
from test.test_sklearn.test_models.mocks import MockClassifier

DataFactory = Callable[..., Tuple[np.ndarray, np.ndarray]]
ENSEMBLE_CLASSES = [SingleClassifierEnsemble, WeightedClassifierEnsemble]


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
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


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
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


@parametrize("classes", [1, 2, 3, ["one", "two"], [[0, 0], [1, 0], [1, 1]]])
@parametrize("ensemble_type", ENSEMBLE_CLASSES)
def test_with_custom_classes_sets_attributes_correctly(
    path: Path,
    ensemble_type: Type[ClassifierEnsemble],
    classes: Union[int, List],
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    make_classifier: Callable[..., MockClassifier],
    make_model_store: Callable[..., ModelStore[MockClassifier]],
) -> None:
    """
    Parameters
    ----------
    path: Path
        The path to create the store at

    ensemble_type: Type[ClassifierEnsemble]
        The type of ensemble to create

    classes: Union[int, List],
        The classes to pass to the ensemble

    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
        Factory to make x, y data

    make_classifier: Callable[..., MockClassifier],
        Factory to make a classifier

    make_model_store: Callable[..., ModelStore[MockClassifier]],
        Factory to make a model store

    ensemble : ClassifierEnsemble
        A fitted ensemble

    Expects
    -------
    * Should set `n_classes_` equivalent to the clf, given the classes of the clf
    * Should set `classes_` equivalent to the clf, given the classes of the clf
    """
    x, y = make_xy("classification", classes=classes)
    store = make_model_store(dir=path)

    keys = [f"m{i}" for i in range(5)]
    for key in keys:
        clf = make_classifier()
        clf.fit(x, y)
        store[key].save(clf)

    clf = store[keys[0]].load()

    ensemble = ensemble_type(model_store=store, classes=clf.classes_)
    ensemble.fit(x, y)

    if isinstance(ensemble.n_classes_, int):
        assert ensemble.n_classes_ == clf.n_classes_
    else:
        np.testing.assert_equal(ensemble.n_classes_, clf.n_classes_)

    np.testing.assert_equal(ensemble.n_classes_, clf.n_classes_)

    # We can't calculate this based off `classes` alone
    # assert ensemble.class_prior_ == clf.class_prior_


@parametrize(
    "classes",
    [
        np.ones(shape=(3, 3, 3)),
        [[[3]]],
    ],
)
@parametrize("ensemble_type", ENSEMBLE_CLASSES)
def test_with_custom_classes_raises_if_wrong_dims(
    path: Path,
    classes: List,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    make_model_store: Callable[..., ModelStore[MockClassifier]],
    ensemble_type: Type[ClassifierEnsemble],
) -> None:
    """
    Parameters
    ----------
    path: Path
        Path to make the model store at

    make_model_store: Callable[..., ModelStore]
        Factory to make a model store

    ensemble : ClassifierEnsemble
        A fitted ensemble

    Expects
    -------
    * Should raise a ValueError if the classes is not the right dimensions
    """
    store = make_model_store(dir=path)
    x, y = make_xy("classification")

    ensemble = ensemble_type(model_store=store, classes=[[[3]]])
    with pytest.raises(ValueError, match="`classes` must be 1 or 2 dimensional"):
        ensemble.fit(x, y)

    return  # pragma: no cover
