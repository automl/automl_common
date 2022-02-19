from typing import Type

from pathlib import Path

from automl_common.backend.stores.store import Store, StoreView

import pytest
import test.test_backend.test_stores.cases as cases
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases


@parametrize_with_cases("cls, dir", cases=cases, filter=ft.has_tag("params"))
def test_construction_ensures_dir_created(cls: Type[StoreView], dir: Path) -> None:
    """
    Parameters
    ----------
    cls: Type[StoreView]
        The Store to construct

    dir: Path
        The path to give it

    kwargs: Union[Dict[str, Any], None]
        Any kwargs to forward

    Expects
    -------
    * To construct without issues
    """
    cls(dir)
    assert dir.exists()


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("populated"))
def test_get_item(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        A store wit populated items

    Expects
    -------
    * Should be able to retrieve each item stored
    """
    for key in store:
        assert store[key] is not None


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("populated"))
def test_contains(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        A store wit populated items

    Expects
    -------
    *
    """
    for key in store:
        assert key in store


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("populated"))
def test_len(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        A store wit populated items

    Expects
    -------
    * The len of the store should be equal to the length of its iterator
    """
    assert len(store) == len(list(iter(store)))


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("populated"))
def test_key_in_path(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        A store wit populated items

    Expects
    -------
    * The key should be part of the "name" part of the path
    """
    for key in list(store):
        assert key in store.path(key).name


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("populated"))
def test_iter_items_have_existing_path(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        A store with populated items

    Expects
    -------
    * Every key gotten from iter should have a path which exists in the store
    """
    for key in store:
        assert store.path(key).exists()


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("populated") & ft.has_tag("store"),
)
def test_set_item(store: Store) -> None:
    """
    Parameters
    ----------
    store: Store
        A store with populated items

    Expects
    -------
    * Should be able to mutate the object and set an item
    * The len should increase by 1 for each item entered
    * The `key` should now be contained in the store
    * The iterator should now contain the `key`
    """
    keys = list(iter(store))
    for key in keys:

        size = len(store)
        new_key = f"{key}_{key}"

        assert new_key not in store
        assert new_key not in iter(store)

        item = store[key]
        store[new_key] = item

        assert new_key in store
        assert new_key in iter(store)
        assert len(store) == size + 1


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("populated") & ft.has_tag("store"),
)
def test_del_item(store: Store) -> None:
    """
    Parameters
    ----------
    store: Store
        A store with populated items

    Expects
    -------
    * Should be able to mutate the object and delete an item
    * The len should decrease by 1 for each item entered
    * The `key` should no longer be contained in the store
    * The iterator should no longer contain the `key`
    """
    keys = list(iter(store))

    for key in keys:
        size = len(store)
        del store[key]

        assert key not in store
        assert key not in iter(store)
        assert len(store) == size - 1


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("unpopulated") & ~ft.has_tag("unstrict_get"),
)
def test_get_item_bad_key(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        An empty store

    Expects
    -------
    * Should get a key error
    """
    key = "badkey"
    with pytest.raises(KeyError):
        store[key]


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("unpopulated"))
def test_contains_bad_key(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        An empty store

    Expects
    -------
    * Should not be contained
    """
    key = "badkey"
    assert key not in store


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("unpopulated"))
def test_len_unpopulated(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        An empty store

    Expects
    -------
    * Should have a length of 0
    """
    assert len(store) == 0


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("unpopulated"))
def test_iter_unpopulated(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        An empty store

    Expects
    -------
    * Should produce no items in it's iter
    """
    assert len(list(iter(store))) == 0


@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("populated"))
def test_load(store: StoreView) -> None:
    """
    Parameters
    ----------
    store: StoreView
        An empty store

    Expects
    -------
    * Should be able to load each item it iterates through
    """
    for key in store:
        assert store.load(key) is not None


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("unpopulated") & ft.has_tag("store"),
)
def test_del_item_bad_key(store: Store) -> None:
    """
    Parameters
    ----------
    store: StoreView
        An empty store

    Expects
    -------
    * Should raise a key error if its no contained
    """
    badkey = "badkey"
    with pytest.raises(KeyError):
        del store[badkey]


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("populated") & ft.has_tag("store"),
)
def test_save(store: Store) -> None:
    """
    Parameters
    ----------
    store : Store
        The store to test

    Expects
    -------
    * Should be able to save an item that was loaded saved
    """
    other_key = "banana-bread"
    key, item = next(iter(store.items()))

    del store[key]

    store[other_key] = item
