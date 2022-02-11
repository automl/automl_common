"""
tags:
    "params" - Provides params to construct a store according to StoreView
    "store" - If the object is a Store as opposed to StoreView
    "populated" - If the store is populated with items
    "unpopulated" - If the store is not populated with items
    "unstrict_get" - If the __getitem__ method doesn't raise a KeyError if file doesn't
        exist. This happens with ModelStore and EnsembleStore.
    "<type>" - The type of store returned
        ["numpy", "predictions", "model", "pickle", "mock_dir"]
"""

from typing import Callable, Tuple, Type, TypeVar

from itertools import chain
from pathlib import Path

import numpy as np
from pytest_cases import case, parametrize

from automl_common.backend.stores.ensemble_store import EnsembleStore
from automl_common.backend.stores.model_store import ModelStore
from automl_common.backend.stores.numpy_store import NumpyStore
from automl_common.backend.stores.pickle_store import PickleStore
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.backend.stores.store import StoreView
from automl_common.ensemble.ensemble import Ensemble
from automl_common.model.model import Model

from test.test_backend.test_stores.mocks import MockDirStore
from test.test_model.mocks import MockModel

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)
# Types used for the construction cases
# For the remaining types, i.e. EnsembleStore
# thos are handled seperatly as they define their own init
PARAMS_CASE_TYPES = [
    ModelStore,
    NumpyStore,
    PredictionsStore,
    PickleStore,
    MockDirStore,
]


@case(id="Already constructed dir", tags=["params"])
@parametrize("cls", PARAMS_CASE_TYPES)
def case_params_with_constructed_dir(
    path: Path,
    cls: Type[StoreView],
) -> Tuple[Type[StoreView], Path]:
    """Types constructed with an already existing dir"""
    return cls, path


@case(id="Not constructed dir", tags=["params"])
@parametrize("cls", PARAMS_CASE_TYPES)
def case_params_with_non_constructed_dir(
    path: Path,
    cls: Type[StoreView],
) -> Tuple[Type[StoreView], Path]:
    """Types constructed with an already existing dir"""
    path = path.joinpath("unconstructed")
    return cls, path


@case(tags=["populated", "store", "numpy"])
def case_numpy_store_populated(
    path: Path,
    make_numpy_store: Callable[..., NumpyStore],
) -> NumpyStore:
    """A populated numpy store"""
    items = {id: np.array([[1]]) for id in "abc"}
    return make_numpy_store(dir=path, items=items)


@case(tags=["unpopulated", "store", "numpy"])
def case_numpy_store_unpopulated(
    path: Path,
    make_numpy_store: Callable[..., NumpyStore],
) -> NumpyStore:
    """An unpopulated numpy store"""
    return make_numpy_store(dir=path)


@case(tags=["populated", "store", "predictions"])
def case_predictions_store_populated(
    path: Path,
    make_predictions_store: Callable[..., PredictionsStore],
) -> PredictionsStore:
    """A populated predictions store"""
    items = {id: np.array([[1]]) for id in "abc"}
    return make_predictions_store(dir=path, items=items)


@case(tags=["unpopulated", "store", "predictions"])
def case_predictions_store_unpopulated(
    path: Path,
    make_predictions_store: Callable[..., PredictionsStore],
) -> PredictionsStore:
    """An unpopulated predictions store"""
    return make_predictions_store(dir=path)


@case(tags=["populated", "unstrict_get", "model"])
def case_model_store_populated(
    path: Path,
    make_model_store: Callable[..., ModelStore[MT]],
    make_model: Callable[..., MT],
) -> ModelStore[MT]:
    """A populated model store"""
    models = {id: make_model() for id in "abc"}
    return make_model_store(dir=path, models=models)


@case(tags=["unpopulated", "unstrict_get", "model"])
def case_model_store_unpopulated(
    path: Path,
    make_model_store: Callable[..., ModelStore[MT]],
    make_model: Callable[..., MT],
) -> ModelStore[MT]:
    """An unpopulated model store"""
    return make_model_store(dir=path)


@case(tags=["populated", "pickle"])
def case_pickle_store_populated(
    path: Path,
    make_pickle_store: Callable[..., PickleStore],
) -> PickleStore:
    """An populated pickle store with pickled 42's"""
    return make_pickle_store(path, {id: 42 for id in "abc"})


@case(tags=["unpopulated", "pickle"])
def case_pickle_store_unpopulated(
    path: Path,
    make_pickle_store: Callable[..., PickleStore],
) -> PickleStore:
    """An unpopulated pickle store"""
    return make_pickle_store(path)


@case(tags=["populated", "unstrict_get", "ensemble"])
@parametrize("model_type", [MockModel])
def case_ensemble_store_populated(
    path: Path,
    model_type: Type[MT],
    make_ensemble_store: Callable[..., EnsembleStore[ET]],
    make_ensemble: Callable[..., ET],
) -> EnsembleStore[ET]:
    """A populated ensemble store"""
    ensemble_dir = path / "ensembles"
    model_dir = path / "models"

    ids_set = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]

    model_store = ModelStore[MT](dir=model_dir)
    ensembles = {
        str(i): make_ensemble(models=ids, model_store=model_store, model_type=model_type)
        for i, ids in enumerate(chain.from_iterable(ids_set))
    }

    return make_ensemble_store(ensemble_dir, ensembles)


@case(tags=["unpopulated", "unstrict_get", "ensemble"])
def case_ensemble_store_unpopulated(
    path: Path,
    make_ensemble_store: Callable,
) -> EnsembleStore:
    """An unpopulated ensemble store"""
    return make_ensemble_store(path)


@case(tags=["unpopulated", "mock_dir", "store"])
def case_mock_dir_unpopoulated(
    path: Path, make_mock_dir_store: Callable[..., MockDirStore]
) -> MockDirStore:
    """An unpopualted MockDirStore"""
    return make_mock_dir_store(path)


@case(tags=["populated", "mock_dir", "store"])
def case_mock_dir_popoulated(
    path: Path, make_mock_dir_store: Callable[..., MockDirStore]
) -> MockDirStore:
    """An unpopualted MockDirStore"""
    return make_mock_dir_store(path, items=dict(zip("abc", "def")))
