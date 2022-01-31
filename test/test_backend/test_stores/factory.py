from typing import Any, Callable, Collection, Mapping, Optional, TypeVar, Union

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

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)


@fixture(scope="function")
def make_predictions_store() -> Callable[..., PredictionsStore]:
    """Make a PredictionsStore

    Parameters
    ----------
    dir: Path
        Path to the store

    items: Optional[Mapping[str, np.ndarray]] = None
        Any {key: np.ndarray} to store

    Returns
    -------
    PredicitonsStore
    """

    def _make(
        dir: Path,
        items: Optional[Mapping[str, np.ndarray]] = None,
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

    items: Optional[Mapping[str, np.ndarray]] = None
        Any {key: np.ndarray} to store

    Returns
    -------
    NumpyStore
    """

    def _make(
        dir: Path,
        items: Optional[Mapping[str, np.ndarray]] = None,
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

    items: Optional[Mapping[str, Any]] = None
        Any {key: item} to store

    """

    def _make(dir: Path, items: Optional[Mapping[str, Any]] = None) -> PickleStore[Any]:
        store = PickleStore[Any](dir=dir)
        if items is not None:
            for key, obj in items.items():
                store[key] = obj

        return store

    return _make


@fixture(scope="function")
def make_model_store() -> Callable[..., ModelStore[MT]]:
    """Make a ModelStore

    Parameters
    ----------
    dir: Path
        Path to the model store

    models: Optional[Mapping[str, Model]] = None
        A dictionary {key: models} to store in the model store

    Returns
    -------
    ModelStore
    """

    def _make(
        dir: Path,
        models: Optional[Mapping[str, MT]] = None,
    ) -> ModelStore[MT]:
        store = ModelStore[MT](dir=dir)
        if models is not None:
            for key, model in models.items():
                store[key].save(model)

        return store

    return _make


@fixture(scope="function")
def make_filtered_model_store() -> Callable[..., FilteredModelStore[MT]]:
    """Make a FilteredModelStore

    Parameters
    ----------
    dir: Path
        Path to the model store

    models: Collection[str] | Mapping[str, Model]
        Mapping of {key: Model} to place in the filtered model store

    extra: Optional[Mapping[str, Model]] = None
        Mapping of {key: Model} to stick in the store outside of filtered model store

    Returns
    -------
    FilteredModelStore
    """

    def _make(
        dir: Path,
        models: Union[Collection[str], Mapping[str, MT]],
        extra: Optional[Mapping[str, MT]] = None,
    ) -> FilteredModelStore[MT]:
        ids = list(models)

        model_store = ModelStore[MT](dir=dir)
        if isinstance(models, Mapping):
            for key, obj in models.items():
                model_store[key].save(obj)

        if extra is not None:
            for key, obj in extra.items():
                model_store[key].save(obj)

        return FilteredModelStore[MT](dir=dir, ids=ids)

    return _make


@fixture(scope="function")
def make_ensemble_store() -> Callable[..., EnsembleStore[ET, MT]]:
    """Make an EnsembleStore

    Parameters
    ----------
    dir: Path
        Path for the EnsembleStore

    model_dir: Path
        Path where Models are stored

    ensembles: Optional[Mapping[str, ET]] = None
        Mapping {key: Ensemble} to store

    extra_models: Optional[Mapping[str, MT]] = None:
        Any extra {key: Model} to store that are outside of the ensemble

    Returns
    -------
    EnsembleStore
    """

    def _make(
        dir: Path,
        model_dir: Path,
        ensembles: Optional[Mapping[str, ET]] = None,
        extra_models: Optional[Mapping[str, MT]] = None,
    ) -> EnsembleStore[ET, MT]:
        store = EnsembleStore[ET, MT](dir=dir, model_dir=model_dir)
        if ensembles is not None:
            for key, ensemble in ensembles.items():
                store[key].save(ensemble)

        if extra_models is not None:
            model_store = ModelStore[MT](dir=model_dir)
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


    items: Optional[Mapping[str, str]] = None
        Mapping {key: items} to store

    Returns
    -------
    MockDirStore
    """

    def _make(dir: Path, items: Optional[Mapping[str, str]] = None) -> MockDirStore:
        store = MockDirStore(dir=dir)
        if items is not None:
            for key, item in items.items():
                store.save(item, key)

        return store

    return _make
