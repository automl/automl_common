"""
tags:
    "params" - If it gives params for constructing an ensemble
    "invalid"/"valid" - If the ensemble is in an invalid/valid state
    <type>+ - mock, uniform, weighted, single
"""
from typing import Callable, TypeVar

from pathlib import Path

from pytest_cases import case

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import (
    Ensemble,
    SingleEnsemble,
    UniformEnsemble,
    WeightedEnsemble,
)
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)


@case(tags=["mock", "valid"])
def case_mock(
    path: Path,
    make_ensemble: Callable[..., MockEnsemble[MT]],
    make_model_store: Callable[..., ModelStore[MT]],
    make_model: Callable[..., MT],
) -> MockEnsemble[MT]:
    """A MockEnsemble with identifiers"""
    store = make_model_store(path)
    ids = ["a", "b", "c"]
    for id in ids:
        store[id].save(make_model())

    return make_ensemble(store, ids)


@case(tags=["single", "valid"])
def case_single(
    path: Path,
    make_single_ensemble: Callable[..., SingleEnsemble[MT]],
    make_model_store: Callable[..., ModelStore[MT]],
    make_model: Callable[..., MT],
) -> SingleEnsemble[MT]:
    """A SingleEnsemble with identifier"""
    store = make_model_store(path)
    store["a"].save(make_model())
    return make_single_ensemble(store, "a")


@case(tags=["uniform", "valid"])
def case_uniform(
    path: Path,
    make_model_store: Callable[..., ModelStore[MT]],
    make_uniform_ensemble: Callable[..., UniformEnsemble[MT]],
    make_model: Callable[..., MT],
) -> UniformEnsemble[MT]:
    """A UniformEnsemble with identifiers"""
    store = make_model_store(path)
    ids = ["a", "b", "c"]
    for id in ids:
        store[id].save(make_model())
    return make_uniform_ensemble(store, ids)


@case(tags=["weighted", "valid"])
def case_weighted(
    path: Path,
    make_weighted_ensemble: Callable[..., WeightedEnsemble[MT]],
    make_model_store: Callable[..., ModelStore[MT]],
    make_model: Callable[..., MT],
) -> WeightedEnsemble[MT]:
    """A WeightedEnsemble with identifiers"""
    store = make_model_store(path)

    ids = ["a", "b", "c"]
    for id in ids:
        store[id].save(make_model())

    weights = {id: 0.3 for id in ids}
    return make_weighted_ensemble(store, weights)
