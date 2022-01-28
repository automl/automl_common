from typing import Callable

from pathlib import Path

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.backend.stores.model_store import FilteredModelStore, ModelStore
from automl_common.model import Model

import test.test_backend.test_stores.cases as cases


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("model") & ft.has_tag("unpopulated"),
)
@parametrize("badkey", ["I_am_a_badkey"])
def test_getitem_with_non_existent_model(store: ModelStore, badkey: str) -> None:
    """
    Parameters
    ----------
    store: ModelStore
        The store to test

    badkey: str
        A badkey non-existnet in the store

    Expects
    -------
    * The badkey should not exist in the store
    * __getitem__ can return an accessor to a model with key, even if there is no
        current folder for it
    """
    assert badkey not in store
    assert store[badkey] is not None


@parametrize("badkey", ["a_bad_model_key"])
@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("filtered_model"))
def test_filtered_model_store_load_with_invalid_key(
    badkey: str,
    store: FilteredModelStore,
) -> None:
    """
    Parameters
    ----------
    badkey: str
        The key to test

    store: FilteredModelStoreStore
        The filtered model store to test

    Expects
    -------
    * A ValueError to be raise as the badkey is not in the identifiers it was
        constructed with
    """
    assert badkey not in store.ids, "Bad test setup"
    with pytest.raises(ValueError):
        store.load(badkey)


@parametrize("badkey", ["a_bad_model_key"])
@parametrize_with_cases("store", cases=cases, filter=ft.has_tag("filtered_model"))
def test_filtered_model_store_getitem_with_invalid_key(
    badkey: str,
    store: FilteredModelStore,
) -> None:
    """
    Parameters
    ----------
    badkey: str
        A key not contained in the filtered_model_store

    store: FilteredModelStore
        The filtered model store to test
    """
    assert badkey not in store.ids, "Bad test setup"
    with pytest.raises(KeyError):
        store[badkey]


def test_filtered_model_store_only_iters_existing_models_in_its_filters(
    path: Path,
    make_filtered_model_store: Callable[..., FilteredModelStore[Model]],
    make_model: Callable[..., Model],
) -> None:
    """
    Parameters
    ----------
    path: Path
        The path to the store

    make_filtered_model_store: (...) -> FilteredModelStore[Model]
        Factory to make a filtered model store

    make_model: (...) -> Model
        Factory to make models

    Expects
    -------
    * The filtered model store should only iterate over models in it's ids
    * Should only iterate over the existing models and not ones that aren't saved
    """
    store = make_filtered_model_store(
        path,
        models={id: make_model() for id in ["a", "b", "c"]},
        extra={id: make_model() for id in ["d", "e", "f"]},
    )

    assert set(store.ids) == {"a", "b", "c"}
    assert set(iter(store)) == {"a", "b", "c"}

    # Delete one
    store["c"].delete(folder=True)

    assert set(store.ids) == {"a", "b", "c"}
    assert set(iter(store)) == {"a", "b"}


def test_filtered_model_store_empty_ids(path: Path) -> None:
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
        FilteredModelStore(dir=path, ids=[])
