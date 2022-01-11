from typing import Iterator, List

import pickle
from pathlib import Path

import numpy as np
from pytest_cases import fixture, parametrize

from automl_common.backend import Backend
from automl_common.backend.contexts import Context
from automl_common.backend.stores import (
    EnsembleStore,
    FilteredModelStore,
    ModelStore,
    NumpyStore,
    PickleStore,
    PredictionsStore,
    Store,
    StoreView,
)
from automl_common.model import Model

from test.test_model.fixtures import MockModel


class MockStoreView(StoreView[int]):
    """For testing StoreView"""

    _mock = {"a": 1, "b": 2, "c": 3}

    def __iter__(self) -> Iterator[str]:
        return iter(self._mock)

    def load(self, key: str) -> int:
        """Dummy load"""
        return self._mock[key]


class MockStore(MockStoreView, Store):
    """For testing Store"""

    def save(self, obj: int, key: str) -> None:
        """Dummy save"""
        self._mock[key] = obj
        with self.path(key).open("w") as f:
            f.write("hello")

    def __delitem__(self, key: str) -> None:
        """Unfortunatly, have to override as we don't know the context"""
        del self._mock[key]
        super().__delitem__(key)


class MockDirStore(MockStore):
    """For testing Store with directory objects"""

    def save(self, obj: int, key: str) -> None:
        """Dummy save"""
        self._mock[key] = obj
        path = self.path(key)
        if not path.exists():
            path.mkdir()


@fixture(scope="function")
def mock_store_view(tmp_path: Path, context: Context) -> MockStoreView:
    """
    Parameters
    ----------
    tmp_path: Path
        The tmp_path to use

    context: Context
        The context to use

    Returns
    -------
    MockStoreView
        A mock version of a StoreView for testing. Will use the context to write
        keys to file
    """
    mock = MockStoreView(dir=tmp_path, context=context)
    for key in mock._mock:
        path = mock.path(key)
        with context.open(path, "w") as f:
            f.write("hello world")

    return mock


@fixture(scope="function")
def mock_store(tmp_path: Path, context: Context) -> MockStore:
    """
    Parameters
    ----------
    tmp_path: Path
        The tmp_path to use

    context: Context
        The context to use

    Returns
    -------
    MockStore
        A mock version of a Store for testing. Will use the context to write
        keys to file
    """
    mock = MockStore(dir=tmp_path, context=context)
    for key in mock._mock:
        path = mock.path(key)
        with context.open(path, "w") as f:
            f.write("hello world")

    return mock


@fixture(scope="function")
def mock_dir_store(tmp_path: Path, context: Context) -> MockDirStore:
    """
    Parameters
    ----------
    tmp_path: Path
        The tmp_path to use

    context: Context
        The context to use

    Returns
    -------
    MockDirStore
        A mock version of a Store for testing. Will use the context to write
        keys as a directory
    """
    mock = MockStore(dir=tmp_path, context=context)
    for key in mock._mock:
        mock.path(key).mkdir()

    return mock


@fixture(scope="function")
def predictions_store(tmp_path: Path, context: Context) -> PredictionsStore:
    """
    Parameters
    ----------
    tmp_path: Path
        The tmp_path to use

    context: Context
        The context to use

    Returns
    -------
    PredictionsStore
        A predictions store for testing comes with 3 predictions saved
        ["train", "test", "val"]
    """
    mock = PredictionsStore(dir=tmp_path, context=context)
    for key in ["train", "test", "val"]:
        path = mock.path(key)
        with context.open(path, "wb") as f:
            np.save(f, np.asarray([1, 1, 1]))

    return mock


@fixture(scope="function")
def numpy_store(tmp_path: Path, context: Context) -> NumpyStore:
    """
    Parameters
    ----------
    tmp_path: Path
        The tmp_path to use

    context: Context
        The context to use

    Returns
    -------
    PredictionsStore
        A predictions store for testing comes with 3 predictions saved
        ["train", "test", "val"]
    """
    mock = NumpyStore(dir=tmp_path, context=context)
    for key in ["a", "b", "c"]:
        path = mock.path(key)
        with path.open("wb") as f:
            np.save(f, np.asarray([1, 1, 1]))

    return mock


@fixture(scope="function")
def model_store_view(
    backend: Backend,
    models: List[Model],
) -> ModelStore[Model]:
    """
    Parameters
    ----------
    backend: Backend
        The backend to use

    models: List[Model]
        A list of models to go in the store view

    Returns
    -------
    ModelStore
        A model_store with three models "model_{0,1,2}"
    """
    store = ModelStore(dir=backend.model_dir, backend=backend)

    for i, model in enumerate(models):
        store[f"model_{i}"].save(model)

    return store


@fixture(scope="function")
def pickle_store(tmp_path: Path, context: Context) -> PickleStore:
    """
    Parameters
    ----------
    tmp_path: Path
        The path to place the store at

    context: Context
        The context to use
    """
    store = PickleStore(dir=tmp_path, context=context)

    objs = {"a": [1, 2], "b": {"hello": "world"}, "c": object()}
    for key, obj in objs.items():
        with store.path(key).open("wb") as f:
            pickle.dump(obj, f)

    return store


@fixture(scope="function")
def filtered_model_store_view(
    backend: Backend,
    models: List[Model],
) -> FilteredModelStore[Model]:
    """
    Parameters
    ----------
    backend: Backend
        The backend to use

    models: List[Model]
        A list of models to go in the store view

    Returns
    -------
    FilteredModelStore
        A model_store with three models "model_{0,1,2}" and ids = [0,2]
    """
    store = ModelStore(dir=backend.model_dir, backend=backend)

    for i, model in enumerate(models):
        store[f"model_{i}"].save(model)

    ids = [f"model_{i}" for i in (0, 2)]
    return FilteredModelStore(dir=backend.model_dir, backend=backend, ids=ids)


@fixture(scope="function")
def ensemble_store_view(tmp_path: Path, backend: Backend) -> EnsembleStore:
    """
    Parameters
    ----------
    tmp_path: Path
        The tmp_path to use

    backend: Backend
        The backend to use

    Returns
    -------
    EnsembleStore
        An EnsembleStore with three ensembles
        ensemble.name = {0,1,2}
    """
    mock = EnsembleStore(dir=tmp_path, backend=backend)
    for i in range(3):
        mock_model = MockModel()
        backend.models[str(i)].save(mock_model)

    return mock


@fixture(scope="function")
@parametrize("impl", [mock_store, mock_dir_store, predictions_store, pickle_store])
def store(impl: Store) -> Store:
    """Accumulate Store's to be tested"""
    return impl


@fixture(scope="function")
@parametrize(
    "impl",
    [
        store,
        mock_store_view,
        model_store_view,
        numpy_store,
        filtered_model_store_view,
        ensemble_store_view,
    ],
)
def store_view(impl: StoreView) -> StoreView:
    """Accumulate StoreView's to be tested"""
    return impl
