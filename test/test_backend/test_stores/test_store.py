from typing import Any, Type, Union

from pathlib import Path

import pytest
from pytest_cases import parametrize

from automl_common.backend.contexts import Context
from automl_common.backend.stores import PredictionsStore, Store, StoreView

from test.test_backend.test_stores.fixtures import MockStore


@parametrize("cls", [MockStore, PredictionsStore])  # noqa
def test_construction(cls: Type[Store], tmp_path: Path, context: Context) -> None:
    """
    Parameters
    ----------
    cls: StoreView
        The Store to construct

    tmp_path: Path
        The path to give it

    context: Context
        The context to give it

    Expects
    -------
    * To construct without issues
    """
    cls(dir=tmp_path, context=context)


@parametrize("cls", [MockStore, PredictionsStore])  # noqa
def test_construction_str_path(
    cls: Type[StoreView],
    tmp_path: Path,
    context: Context,
) -> None:
    """
    Parameters
    ----------
    cls: Store
        The Store to construct

    tmp_path: Path
        The path to give it, converted to str

    context: Context
        The context to give it

    Expects
    -------
    * It's internal reference is to a Path object
    """
    path = str(tmp_path)
    store = cls(dir=path, context=context)

    assert isinstance(store.dir, Path)


@parametrize("cls", [MockStore, PredictionsStore])  # noqa
def test_construction_non_existing_path(
    cls: Type[Store],
    tmp_path: Path,
    context: Context,
) -> None:
    """
    Parameters
    ----------
    cls: Store
        The Store to construct

    tmp_path: Path
        The path from which to make a new dir

    context: Context
        The context to give it

    Expects
    -------
    * It's internal reference is to a Path object
    """
    path = tmp_path / "hello"
    cls(dir=path, context=context)

    assert path.exists()


def test_get_item(store_view: StoreView) -> None:
    """
    Parameters
    ----------
    store_view: StoreView
        The store_view to test

    Expects
    -------
    * Expects keys in iteratr to be able to be retreived without issue
    """
    for key in store_view:
        store_view.__getitem__(key)
        store_view[key]


def test_contains(store_view: StoreView) -> None:
    """
    Parameters
    ----------
    store_view: StoreView
        The store_view to test

    Expects
    -------
    * Expects all keys advertised through iter will be contained
    """
    assert all(key in store_view for key in iter(store_view))


@parametrize("bad_key", ["invalid_key1", "invalid_key2", 42, object()])
def test_contains_with_invalid_key(
    store_view: StoreView,
    bad_key: Union[str, Any],
) -> None:
    """
    Parameters
    ----------
    store_view: StoreView
        The store_view to test

    bad_key: Union[str, Any]
        A bad key that is not contained or non-strings

    Expects
    -------
    * Key should not be contained, returning False
    """
    assert bad_key not in store_view


def test_len(store_view: StoreView) -> None:
    """
    Parameters
    ----------
    store_view: StoreView
        The store_view to test

    Expects
    -------
    * The len of a store view should be the length of its iter
    """
    assert len(store_view) == len(list(iter(store_view)))


def test_path(store_view: StoreView) -> None:
    """
    Parameters
    ----------
    store_view: StoreView
        The store_view to test

    Expects
    -------
    * The path should be a Path object
    * It should have the key in the end of the path
    * It should contain the dir in the base of the path
    """
    path = store_view.path("key")

    assert isinstance(path, Path)
    assert "key" in str(path.name)
    assert str(store_view.dir) in str(path.parent)


def test_iter(store_view: StoreView) -> None:
    """
    Parameters
    ----------
    store_view: StoreView
        The store_view to test

    Expects
    -------
    * Should be able to iterate through store_view multiple times and recieve
        same result
    """
    assert list(store_view) == list(store_view.__iter__())


def test_setitem(store: Store) -> None:
    """
    Parameters
    ----------
    store: Store
        The store to test

    Expects
    -------
    * Should save without issue
    * The saved item should not be contained
    * The saved item should show up when iterating
    """
    # We do this generically by loading an item, and saving it under a new key
    key = next(iter(store))
    obj = store[key]

    del store[key]

    new_key = f"{key}_{key}"
    store[new_key] = obj

    assert new_key in store
    assert new_key in list(iter(store))


def test_del(store: Store) -> None:
    """
    Parameters
    ----------
    store: Store
        The store to test

    Expects
    -------
    * Deleting an item should mean it no longer is retrievable, contained or
        exists at the path
    """
    keys = list(store)
    for key in keys:

        assert store[key] is not None
        assert key in store
        assert store.path(key).exists()

        del store[key]

        with pytest.raises((KeyError, FileNotFoundError)):
            store[key]
        assert key not in store
        assert not store.path(key).exists()
