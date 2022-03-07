from typing import Any, Callable, Mapping, Optional, TypeVar

from pathlib import Path

from pytest_cases import fixture

import numpy as np

from automl_common.backend.stores.ensemble_store import EnsembleStore
from automl_common.backend.stores.model_store import ModelStore
from automl_common.backend.stores.numpy_store import NumpyStore
from automl_common.backend.stores.pickle_store import PickleStore
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.ensemble import Ensemble
from automl_common.model import Model

from test.test_backend.test_stores.mocks import MockDirStore

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)
ID = TypeVar("ID")


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
        items: Optional[Mapping[ID, np.ndarray]] = None,
    ) -> NumpyStore:
        store = NumpyStore[ID](dir=dir)
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

    items: Optional[Mapping[ID, Any]] = None
        Any {key: item} to store

    """

    def _make(dir: Path, items: Optional[Mapping[ID, Any]] = None) -> PickleStore[ID, Any]:
        store = PickleStore[ID, Any](dir=dir)
        if items is not None:
            for key, obj in items.items():
                store[key] = obj

        return store

    return _make


@fixture(scope="function")
def make_model_store() -> Callable[..., ModelStore[ID, MT]]:
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
        models: Optional[Mapping[ID, MT]] = None,
    ) -> ModelStore[ID, MT]:
        # Make a store unrestricted by ids
        store = ModelStore[ID, MT](dir=dir)
        if models is not None:
            for key, model in models.items():
                store[key].save(model)

        return ModelStore(dir=dir)

    return _make


@fixture(scope="function")
def make_ensemble_store() -> Callable[..., EnsembleStore[ID, ET]]:
    """Make an EnsembleStore

    Parameters
    ----------
    dir: Path
        Path for the EnsembleStore

    model_dir: Path
        Path where Models are stored

    ensembles: Optional[Mapping[ID, ET]] = None
        Mapping {key: Ensemble} to store

    extra_models: Optional[Mapping[ID, MT]] = None:
        Any extra {key: Model} to store that are outside of the ensemble

    Returns
    -------
    EnsembleStore
    """

    def _make(
        dir: Path,
        ensembles: Optional[Mapping[ID, ET]] = None,
    ) -> EnsembleStore[ID, ET]:
        store = EnsembleStore[ID, ET](dir=dir)
        if ensembles is not None:
            for key, ensemble in ensembles.items():
                store[key].save(ensemble)

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
