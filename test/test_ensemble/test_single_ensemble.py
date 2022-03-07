from typing import Callable, TypeVar

from pathlib import Path

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import SingleEnsemble
from automl_common.model import Model

import test.test_ensemble.cases as cases

MT = TypeVar("MT", bound=Model)


def test_empty_model_id(path: Path, make_model_store: Callable[..., ModelStore]) -> None:
    """
    Parameters
    ----------
    path : Path
        A dummy path to use

    Expects
    -------
    * Should raise a value error if the model_id is an empty string
    """
    store = make_model_store(path)
    with pytest.raises(ValueError):
        SingleEnsemble(store, model_id="")


@parametrize_with_cases("single_ensemble", cases=cases, filter=ft.has_tag("single"))
def test_single_ensemble_model(single_ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    single_ensemble: SingleEnsemble[MT]
        SingleEnsemble with a saved model

    Expects
    -------
    * Should be able to access single model
    """
    assert single_ensemble.model is not None


@parametrize_with_cases("single_ensemble", cases=cases, filter=ft.has_tag("single"))
def test_has_length_of_one_with_full_weight(single_ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    single_ensemble: SingleEnsemble
        SingleEnsemble with a saved model

    Expects
    -------
    * Should have a length of one with a weight of 1.0
    """
    assert single_ensemble.weights == {single_ensemble.model_id: 1.0}
