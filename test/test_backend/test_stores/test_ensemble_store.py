from pathlib import Path

from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.backend.stores.ensemble_store import EnsembleStore

import test.test_backend.test_stores.cases as cases


def test_creates_model_dir(path: Path) -> None:
    """
    Parameters
    ----------
    path: Path
        The path to base from

    Expects
    -------
    * Should create the models dir if it doesn't already exist
    """
    ensemble_dir = path / "ensembles"
    model_dir = path / "models"
    EnsembleStore(ensemble_dir, model_dir=model_dir)


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
