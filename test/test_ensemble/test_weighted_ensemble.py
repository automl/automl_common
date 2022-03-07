from typing import Callable, TypeVar

from pathlib import Path
from unittest.mock import patch

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

import numpy as np

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import WeightedEnsemble
from automl_common.model import Model

import test.test_ensemble.cases as cases

MT = TypeVar("MT", bound=Model)


def test_empty_ids(path: Path, make_model_store: Callable[..., ModelStore]) -> None:
    """
    Parameters
    ----------
    path : Path
        A dummy path to use

    Expects
    -------
    * Should raise a value error if the ids is empty
    """
    store = make_model_store(path)
    with pytest.raises(ValueError):
        WeightedEnsemble(store, weighted_ids={})

    return  # pragma: no cover


def test_with_missing_models(path: Path, make_model_store: Callable[..., ModelStore]) -> None:
    """
    Parameters
    ----------
    path : Path
        Path to place the model store at

    make_model_store : Callable[..., ModelStore[MT]]
        Factory to make a model store

    Expects
    -------
    * Should raise an error if constructed with ids of models not in the store
    """
    store = make_model_store(path)
    badkey = "bad"
    with pytest.raises(ValueError):
        WeightedEnsemble(store, weighted_ids={badkey: 1.0})

    return  # pragma: no cover


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted"),
)
def test_predict(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble: WeightedEnsemble
        A weighted ensemble with weights

    Expects
    -------
    * Weighted_sum function should recieve the weights in the correct order with the
      predictions.
    """
    x = np.asarray([1, 10, 100])

    ids, weights = zip(*ensemble.weights.items())

    # Note sure why coverage gives `line -> exit` not covered
    expected = list(iter(ensemble[id].predict(x) for id in ids))  # pragma: no cover

    with patch("automl_common.ensemble.weighted_ensemble.weighted_sum") as mock:
        ensemble.predict(x)

        args, kwargs = mock.call_args
        assert list(weights) == list(kwargs["weights"])

        for input, expected in zip(args[0], expected):
            np.testing.assert_equal(input, expected)

    return
