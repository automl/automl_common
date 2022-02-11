from typing import Callable, Dict, Mapping, Optional, TypeVar, Union
from typing_extensions import Literal  # TODO, python 3.8

from pathlib import Path

import numpy as np
from pytest_cases import fixture

from automl_common.backend.stores.model_store import ModelStore
from automl_common.metrics import accuracy, rmse
from automl_common.sklearn.ensemble import (
    SingleClassifierEnsemble,
    SingleRegressorEnsemble,
    WeightedClassifierEnsemble,
    WeightedRegressorEnsemble,
)
from automl_common.sklearn.model import Classifier, Estimator, Regressor
from automl_common.util.types import Orderable

from test.data import DEFAULT_SEED, XYPack, xy
from test.test_sklearn.test_models.mocks import MockClassifier, MockRegressor

EstimatorT = TypeVar("EstimatorT", bound=Estimator)
ClassifierT = TypeVar("ClassifierT", bound=Classifier)
RegressorT = TypeVar("RegressorT", bound=Regressor)


data_clf = XYPack(*xy(kind="classification", xdims=(50, 3), ydims=(50,)))
data_rgr = XYPack(*xy(kind="regression", xdims=(50, 3), ydims=(50,)))


@fixture(scope="function")
def make_sklearn_weighted_classifier_ensemble() -> Callable[
    ..., WeightedClassifierEnsemble[ClassifierT]
]:
    """Make a WeightedClassifierEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[ClassifierT]] = None,
        size: int = 10,
        fitted: bool = False,
        voting: Literal["majority", "probability"] = "majority",
        x: np.ndarray = data_clf.x,
        y: np.ndarray = data_clf.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, ClassifierT]] = 10,
        model_x: np.ndarray = data_clf.x,
        model_y: np.ndarray = data_clf.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> WeightedClassifierEnsemble[ClassifierT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[ClassifierT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, ClassifierT] = {
                str(i): MockClassifier(seed=seed) for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)
            store[name].save(model)

        ensemble = WeightedClassifierEnsemble[ClassifierT](
            model_store=store,
            size=size,
            random_state=seed,
            metric=metric,
            select=select,
            voting=voting,
        )

        if fitted:
            ensemble.fit(x, y)

        return ensemble

    return _make


@fixture(scope="function")
def make_sklearn_weighted_regressor_ensemble() -> Callable[
    ..., WeightedRegressorEnsemble[RegressorT]
]:
    """Make a WeightedRegressorEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[RegressorT]] = None,
        size: int = 10,
        fitted: bool = False,
        x: np.ndarray = data_clf.x,
        y: np.ndarray = data_clf.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, RegressorT]] = 10,
        model_x: np.ndarray = data_clf.x,
        model_y: np.ndarray = data_clf.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> WeightedRegressorEnsemble[RegressorT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[RegressorT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, RegressorT] = {
                str(i): MockRegressor() for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)

            store[name].save(model)

        ensemble = WeightedRegressorEnsemble[RegressorT](
            model_store=store,
            size=size,
            random_state=seed,
            metric=metric,
            select=select,
        )

        if fitted:
            ensemble.fit(x, y)

        return ensemble

    return _make


@fixture(scope="function")
def make_sklearn_single_classifier_ensemble() -> Callable[
    ..., SingleClassifierEnsemble[ClassifierT]
]:
    """Make a SingleClassifierEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[ClassifierT]] = None,
        fitted: bool = False,
        x: np.ndarray = data_clf.x,
        y: np.ndarray = data_clf.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, ClassifierT]] = 10,
        model_x: np.ndarray = data_clf.x,
        model_y: np.ndarray = data_clf.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> SingleClassifierEnsemble[ClassifierT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[ClassifierT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, ClassifierT] = {
                str(i): MockClassifier(seed=seed) for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)
            store[name].save(model)

        ensemble = SingleClassifierEnsemble[ClassifierT](
            model_store=store,
            random_state=seed,
            metric=metric,
            select=select,
        )

        if fitted:
            ensemble.fit(x, y)

        return ensemble

    return _make


@fixture(scope="function")
def make_sklearn_single_regressor_ensemble() -> Callable[..., SingleRegressorEnsemble[RegressorT]]:
    """Make a SingleRegressorEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[RegressorT]] = None,
        fitted: bool = False,
        x: np.ndarray = data_clf.x,
        y: np.ndarray = data_clf.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, RegressorT]] = 10,
        model_x: np.ndarray = data_clf.x,
        model_y: np.ndarray = data_clf.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> SingleRegressorEnsemble[RegressorT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[RegressorT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict : Dict[str, RegressorT] = {
                str(i): MockRegressor() for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)
            store[name].save(model)

        ensemble = SingleRegressorEnsemble[RegressorT](
            model_store=store,
            random_state=seed,
            metric=metric,
            select=select,
        )

        if fitted:
            ensemble.fit(x, y)

        return ensemble

    return _make
