from typing import Callable, TypeVar

from pathlib import Path

import pytest
from pytest_cases import case, parametrize, parametrize_with_cases

from automl_common.backend.stores.model_store import ModelStore
from automl_common.model import Model

ID = TypeVar("ID")


@case(tags=["filter"])
def case_model_store_with_ids(
    path: Path,
    make_model_store: Callable[..., ModelStore[ID, Model]],
    make_model: Callable[..., Model],
) -> ModelStore[ID, Model]:
    store = make_model_store(
        path,
        models={id: make_model() for id in ["a", "b", "c", "d", "e", "f"]},
        ids=["a", "b", "c"],
    )
    return store


@case(tags=["no_filter"])
def case_model_store(
    path: Path,
    make_model_store: Callable[..., ModelStore[ID, Model]],
    make_model: Callable[..., Model],
) -> ModelStore[ID, Model]:
    store = make_model_store(
        path,
        models={id: make_model() for id in ["a", "b", "c", "d", "e", "f"]},
    )
    return store


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
