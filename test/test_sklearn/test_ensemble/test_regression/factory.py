from typing import Callable, Dict, Mapping, Optional, TypeVar, Union
from typing_extensions import Literal  # TODO, python 3.8

from functools import partial
from pathlib import Path

from pytest_cases import fixture

import numpy as np
from sklearn.metrics import mean_squared_error

from automl_common.backend.stores.model_store import ModelStore
from automl_common.sklearn.ensemble.regression import (
    SingleRegressorEnsemble,
    WeightedRegressorEnsemble,
)
from automl_common.sklearn.model import Regressor
from automl_common.util.types import Orderable

from test.data import DEFAULT_SEED, XYPack, xy
from test.test_sklearn.test_models.mocks import MockRegressor

RT = TypeVar("RT", bound=Regressor)
rmse = partial(mean_squared_error, squared=False)


data = XYPack(*xy(kind="regression", xdims=(50, 3), ydims=(50,)))


@fixture(scope="function")
def make_sklearn_weighted_regressor_ensemble() -> Callable[..., WeightedRegressorEnsemble[RT]]:
    """Make a WeightedRegressorEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[RT]] = None,
        size: int = 10,
        fitted: bool = False,
        x: np.ndarray = data.x,
        y: np.ndarray = data.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, RT]] = 10,
        model_x: np.ndarray = data.x,
        model_y: np.ndarray = data.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> WeightedRegressorEnsemble[RT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[RT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, RT] = {
                str(i): MockRegressor() for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)

            store[name].save(model)

        ensemble = WeightedRegressorEnsemble[RT](
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
def make_sklearn_single_regressor_ensemble() -> Callable[..., SingleRegressorEnsemble[RT]]:
    """Make a SingleRegressorEnsemble"""

    def _make(
        path: Optional[Path] = None,
        model_store: Optional[ModelStore[RT]] = None,
        fitted: bool = False,
        x: np.ndarray = data.x,
        y: np.ndarray = data.y,
        metric: Callable[[np.ndarray, np.ndarray], Orderable] = rmse,
        select: Literal["min", "max"] = "min",
        random_state: Optional[Union[int, np.random.RandomState]] = DEFAULT_SEED,
        models: Union[int, Mapping[str, RT]] = 10,
        model_x: np.ndarray = data.x,
        model_y: np.ndarray = data.y,
        fit_models: bool = True,
        seed: Union[int, np.random.RandomState] = DEFAULT_SEED,
    ) -> SingleRegressorEnsemble[RT]:
        # Get model store
        if model_store is not None:
            store = model_store
        elif path is not None:
            store = ModelStore[RT](dir=path)
        else:
            raise NotImplementedError()

        # Build models
        if isinstance(models, int):
            model_dict: Dict[str, RT] = {
                str(i): MockRegressor() for i in range(models)  # type: ignore
            }
        else:
            model_dict = dict(**models)

        # Put models in it
        for name, model in model_dict.items():
            if fit_models:
                model.fit(model_x, model_y)
            store[name].save(model)

        ensemble = SingleRegressorEnsemble[RT](
            model_store=store,
            random_state=seed,
            metric=metric,
            select=select,
        )

        if fitted:
            ensemble.fit(x, y)

        return ensemble

    return _make
