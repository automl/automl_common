from typing import Callable, TypeVar

from math import isclose
from pathlib import Path

import pytest
from pytest_cases import parametrize_with_cases

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import UniformEnsemble
from automl_common.model import Model

import test.test_ensemble.cases as cases

MT = TypeVar("MT", bound=Model)


def test_empty_ids(
    path: Path,
    make_model_store: Callable[..., ModelStore[MT]],
) -> None:
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
        UniformEnsemble(store, ids=[])


@parametrize_with_cases("uniform_ensemble", cases=cases, has_tag="uniform")
def test_uniform_ensemble_distributes_weight(uniform_ensemble: UniformEnsemble) -> None:
    """
    Parameters
    ----------
    uniform_ensemble: UniformEnsemble
        UniformEnsemble with n saved model

    Expects
    -------
    * Should have an equal weight across all models
    * Weights should add up to 1
    * Each weight should be 1/n
    """
    weights = list(uniform_ensemble.weights.values())

    assert all(w == weights[0] for w in weights)

    assert isclose(sum(weights), 1)

    expected = 1.0 / len(weights)
    assert all(isclose(w, expected) for w in weights)
