from typing import Callable, Dict, Mapping, Optional, TypeVar, Union
from typing_extensions import Literal  # TODO, python 3.8

from pathlib import Path

import numpy as np
from pytest_cases import fixture

from automl_common.backend.stores.model_store import ModelStore
from automl_common.metrics import accuracy
from automl_common.sklearn.ensemble.classification import (
    SingleClassifierEnsemble,
    WeightedClassifierEnsemble,
)
from automl_common.sklearn.model import Classifier
from automl_common.util.types import Orderable

from test.data import DEFAULT_SEED, XYPack, xy
from test.test_sklearn.test_models.mocks import MockClassifier

CT = TypeVar("CT", bound=Classifier)


data = XYPack(*xy(kind="classification", xdims=(50, 3), ydims=(50,)))


@fixture(scope="function")
def make_sklearn_weighted_classifier_ensemble() -> Callable[..., WeightedClassifierEnsemble[CT]]:
    """Make a WeightedClassifierEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[CT]] = None,
        size: int = 10,
        fitted: bool = False,
        voting: Literal["majority", "probability"] = "probability",
        x: np.ndarray = data.x,
        y: np.ndarray = data.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, CT]] = 10,
        model_x: np.ndarray = data.x,
        model_y: np.ndarray = data.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> WeightedClassifierEnsemble[CT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[CT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, CT] = {
                str(i): MockClassifier() for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)
            store[name].save(model)

        ensemble = WeightedClassifierEnsemble[CT](
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
def make_sklearn_single_classifier_ensemble() -> Callable[..., SingleClassifierEnsemble[CT]]:
    """Make a SingleClassifierEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[CT]] = None,
        fitted: bool = False,
        x: np.ndarray = data.x,
        y: np.ndarray = data.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = accuracy,
        select: Literal["min", "max"] = "max",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, CT]] = 10,
        model_x: np.ndarray = data.x,
        model_y: np.ndarray = data.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> SingleClassifierEnsemble[CT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[CT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, CT] = {
                str(i): MockClassifier() for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)
            store[name].save(model)

        ensemble = SingleClassifierEnsemble[CT](
            model_store=store,
            random_state=seed,
            metric=metric,
            select=select,
        )

        if fitted:
            ensemble.fit(x, y)

        return ensemble

    return _make
