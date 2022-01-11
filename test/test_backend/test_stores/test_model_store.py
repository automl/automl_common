import pytest
from pytest_cases import parametrize

from automl_common.backend.stores.model_store import FilteredModelStore


@parametrize("badkey", ["a_bad_model_key"])
def test_load_with_invalid_key(
    filtered_model_store_view: FilteredModelStore,
    badkey: str,
) -> None:
    """
    Parameters
    ----------
    filtered_model_store: FilteredModelStore
        The store to test

    badkey: str
        A bad key not contained
    """
    with pytest.raises(ValueError):
        filtered_model_store_view.load(badkey)
