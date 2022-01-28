from typing import Any, Callable, Collection, Dict, Mapping, Optional, TypeVar, Union

from pathlib import Path

import numpy as np
from pytest_cases import fixture

from automl_common.backend.stores.ensemble_store import EnsembleStore
from automl_common.backend.stores.model_store import FilteredModelStore, ModelStore
from automl_common.backend.stores.numpy_store import NumpyStore
from automl_common.backend.stores.pickle_store import PickleStore
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.ensemble import Ensemble
from automl_common.model import Model

from test.test_backend.test_stores.mocks import MockDirStore

ModelT = TypeVar("ModelT", bound=Model)


@fixture(scope="function")
def make_predictions_store() -> Callable[..., PredictionsStore]:
    """Make a PredictionsStore

    Parameters
    ----------
    dir: Path
        Path to the store

    items: Optional[Dict[str, np.ndarray]] = None
        Any {key: np.ndarray} to store

    Returns
    -------
    PredicitonsStore
    """

    def _make(
        dir: Path,
        items: Optional[Dict[str, np.ndarray]] = None,
    ) -> PredictionsStore:
        store = PredictionsStore(dir=dir)
        if items is not None:
            for key, arr in items.items():
                store[key] = arr

        return store

    return _make


@fixture(scope="function")
def make_numpy_store() -> Callable[..., NumpyStore]:
    """Make a NumpyStore

    Parameters
    ----------
    dir: Path
        Path to the store

    items: Optional[Dict[str, np.ndarray]] = None
        Any {key: np.ndarray} to store

    Returns
    -------
    NumpyStore
    """

    def _make(
        dir: Path,
        items: Optional[Dict[str, np.ndarray]] = None,
    ) -> NumpyStore:
        store = NumpyStore(dir=dir)
        if items is not None:
            for key, arr in items.items():
                store[key] = arr

        return store

    return _make


@fixture(scope="function")
def make_pickle_store() -> Callable[..., PickleStore]:
    """Make a PickleStore

    Parameters
    ----------
    dir: Path
        Path to the store

    items: Optional[Dict[str, Any]] = None
        Any {key: item} to store

    """

    def _make(dir: Path, items: Optional[Dict[str, Any]] = None) -> PickleStore[Any]:
        store = PickleStore[Any](dir=dir)
        if items is not None:
            for key, obj in items.items():
                store[key] = obj

        return store

    return _make


@fixture(scope="function")
def make_model_store() -> Callable[..., ModelStore[ModelT]]:
    """Make a ModelStore

    Parameters
    ----------
    dir: Path
        Path to the model store

    models: Optional[Dict[str, Model]] = None
        A dictionary {key: models} to store in the model store

    Returns
    -------
    ModelStore
    """

    def _make(
        dir: Path,
        models: Optional[Dict[str, ModelT]] = None,
    ) -> ModelStore[ModelT]:
        store = ModelStore[ModelT](dir=dir)
        if models is not None:
            for key, model in models.items():
                store[key].save(model)

        return store

    return _make


@fixture(scope="function")
def make_filtered_model_store() -> Callable[..., FilteredModelStore[ModelT]]:
    """Make a FilteredModelStore

    Parameters
    ----------
    dir: Path
        Path to the model store

    models: Dict[str, Model]
        Dictionary of {key: Model} to place in the filtered model store

    extra: Optional[Dict[str, Model]] = None
        Dictionary of {key: Model} to stick in the store outside of filtered model store

    Returns
    -------
    FilteredModelStore
    """

    def _make(
        dir: Path,
        models: Union[Collection[str], Dict[str, ModelT]],
        extra: Optional[Dict[str, ModelT]] = None,
    ) -> FilteredModelStore[ModelT]:
        model_store = ModelStore[ModelT](dir=dir)

        if isinstance(models, Mapping):
            ids = list(models.keys())
            for key, obj in models.items():
                model_store[key].save(obj)
        else:
            ids = models

        if extra is not None:
            for key, obj in extra.items():
                model_store[key].save(obj)

        return FilteredModelStore[ModelT](dir=dir, ids=ids)

    return _make


@fixture(scope="function")
def make_ensemble_store() -> Callable[..., EnsembleStore[ModelT]]:
    """Make an EnsembleStore

    Parameters
    ----------
    dir: Path
        Path for the EnsembleStore

    model_dir: Path
        Path where Models are stored

    ensembles: Optional[Dict[str, Ensemble]] = None
        Dictionary {key: Ensemble} to store

    extra_models: Optional[Dict[str, Model]] = None:
        Any extra {key: Model} to store that are outside of the ensemble

    Returns
    -------
    EnsembleStore
    """

    def _make(
        dir: Path,
        model_dir: Path,
        ensembles: Optional[Dict[str, Ensemble[ModelT]]] = None,
        extra_models: Optional[Dict[str, ModelT]] = None,
    ) -> EnsembleStore[ModelT]:
        store = EnsembleStore[ModelT](dir=dir, model_dir=model_dir)
        if ensembles is not None:
            for key, ensemble in ensembles.items():
                store[key].save(ensemble)

        if extra_models is not None:
            model_store = ModelStore[ModelT](dir=model_dir)
            for key, model in extra_models.items():
                model_store[key].save(model)

        return store

    return _make


@fixture(scope="function")
def make_mock_dir_store() -> Callable[..., MockDirStore]:
    """Make a MockDirStore

    Parameters
    ----------
    dir: Path
        Path for the EnsembleStore


    items: Optional[Dict[str, str]] = None
        Dictionary {key: items} to store

    Returns
    -------
    MockDirStore
    """

    def _make(
        dir: Path, items: Optional[Mapping[str, str]] = None
    ) -> EnsembleStore[ModelT]:
        store = MockDirStore(dir=dir)
        if items is not None:
            for key, item in items.items():
                store.save(item, key)

        return store

    return _make
