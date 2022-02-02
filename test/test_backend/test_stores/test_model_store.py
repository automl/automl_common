from typing import Callable

from pathlib import Path

import pytest
from pytest_cases import case, parametrize, parametrize_with_cases

from automl_common.backend.stores.model_store import ModelStore
from automl_common.model import Model


@case(tags=["filter"])
def case_model_store_with_ids(
    path: Path,
    make_model_store: Callable[..., ModelStore[Model]],
    make_model: Callable[..., Model],
) -> ModelStore[Model]:
    store = make_model_store(
        path,
        models={id: make_model() for id in ["a", "b", "c", "d", "e", "f"]},
        ids=["a", "b", "c"],
    )
    return store


@case(tags=["no_filter"])
def case_model_store(
    path: Path,
    make_model_store: Callable[..., ModelStore[Model]],
    make_model: Callable[..., Model],
) -> ModelStore[Model]:
    store = make_model_store(
        path,
        models={id: make_model() for id in ["a", "b", "c", "d", "e", "f"]},
    )
    return store


def test_construction_with_empty_ids(path: Path) -> None:
    """
    Parameters
    ----------
    path: Path
        Path to the store

    Expects
    -------
    * Should raise a ValueError as we should not have no ids as the filter ids
    """
    with pytest.raises(ValueError):
        ModelStore(dir=path, ids=[])


@parametrize_with_cases("store", cases=".", has_tag="filter")
def test_iter_with_filter(store: ModelStore[Model]) -> None:
    """
    Parameters
    ----------
    store: ModelStore[Model]
        A store with an id filter

    Expects
    -------
    * The filtered model store should only iterate over models in it's ids and not
    all models in the directory
    * The length of a filtered model store is less than a store made in the same dir
    """
    assert store.ids is not None

    unrestricted_store = ModelStore[Model](dir=store.dir)

    assert set(store).issubset(unrestricted_store)
    assert len(store) < len(unrestricted_store)


@parametrize_with_cases("store", cases=".", has_tag="no_filter")
@parametrize("badkey", ["this_is_a_badkey"])
def test_getitem_not_strict(store: ModelStore, badkey: str) -> None:
    """
    Parameters
    ----------
    store: ModelStore
        The store to test

    badkey: str
        A badkey non-existing in the store

    Expects
    -------
    * The badkey should not exist in the store
    * __getitem__ can return an accessor to a model with key, even if there is no
        current folder for it
    """
    assert badkey not in store
    assert store[badkey] is not None


@parametrize("badkey", ["a_bad_model_key"])
@parametrize_with_cases("store", cases=".", has_tag="filter")
def test_getitem_strict_with_filter(badkey: str, store: ModelStore) -> None:
    """
    Parameters
    ----------
    badkey: str
        A key not contained in the filtered_model_store

    store: ModelStore
        The filtered model store to test

    Expects
    -------
    * Should not be able to __getitem__ which does not exist in it's ids
    """
    with pytest.raises(KeyError, match=f"{badkey} not in identifiers"):
        store[badkey]


@parametrize("badkey", ["this_is_a_badkey"])
@parametrize_with_cases("store", cases=".", has_tag="filter")
def test_load_with_filter(badkey: str, store: ModelStore) -> None:
    """
    Parameters
    ----------
    badkey: str
        The key to test

    store: ModelStore
        The model store with and ids filter

    Expects
    -------
    * A ValueError to be raise as the badkey is not in the identifiers it was
        constructed with
    """
    with pytest.raises(ValueError, match=f"{badkey} not in identifiers"):
        store.load(badkey)
