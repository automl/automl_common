from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.backend.stores.ensemble_store import EnsembleStore

import test.test_backend.test_stores.cases as cases


@parametrize_with_cases(
    "store",
    cases=cases,
    filter=ft.has_tag("ensemble") & ft.has_tag("unpopulated"),
)
@parametrize("key", ["non_existent_key"])
def test_get_item_with_non_existent_ensemble(store: EnsembleStore, key: str) -> None:
    """
    Parameters
    ----------
    store: EnsembleStore
        The EnsembleStore to test

    key: str
        The key to load

    Expects
    -------
    * The key should not exist in the store
    * ...yet it should still allowed be used in __getitem__
    """
    assert key not in store
    assert store[key] is not None
