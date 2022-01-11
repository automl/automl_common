from typing import Iterator

from pathlib import Path

import numpy as np
from pytest_cases import fixture, parametrize

from automl_common.backend.contexts import Context
from automl_common.backend.stores import PredictionsStore, Store, StoreView


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
        with self.context.open(self.path(key), "w") as f:
            f.write("hello")

    def __delitem__(self, key: str) -> None:
        """Unfortunatly, have to override as we don't know the context"""
        del self._mock[key]
        super().__delitem__(key)


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
@parametrize("impl", [mock_store_view, predictions_store])
def store_view(impl: StoreView) -> StoreView:
    """Accumulate StoreView's to be tested"""
    return impl


@fixture(scope="function")
@parametrize("impl", [mock_store, predictions_store])
def store(impl: Store) -> Store:
    """Accumulate Store's to be tested"""
    return impl
