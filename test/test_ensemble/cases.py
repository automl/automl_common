"""
tags:
    "params" - If it gives params for constructing an ensemble
    "invalid"/"valid" - If the ensemble is in an invalid/valid state
    <type>+ - mock, uniform, weighted, single
"""
from typing import Callable

from pathlib import Path

from pytest_cases import case

from automl_common.ensemble import SingleEnsemble, UniformEnsemble, WeightedEnsemble
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble


@case(tags=["mock", "valid"])
def case_mock(
    path: Path,
    make_ensemble: Callable,
    make_model: Callable,
) -> MockEnsemble:
    """A MockEnsemble with identifiers"""
    return make_ensemble(path, {id: make_model() for id in "abc"})


@case(tags=["single", "valid"])
def case_single(
    path: Path,
    make_single_ensemble: Callable,
    make_model: Callable,
) -> SingleEnsemble[Model]:
    """A SingleEnsemble with identifier"""
    return make_single_ensemble(path, "a", make_model())


@case(tags=["uniform", "valid"])
def case_uniform(
    path: Path,
    make_uniform_ensemble: Callable,
    make_model: Callable,
) -> UniformEnsemble[Model]:
    """A UniformEnsemble with identifiers"""
    return make_uniform_ensemble(path, {id: make_model() for id in "abc"})


@case(tags=["weighted", "valid"])
def case_weighted(
    path: Path,
    make_weighted_ensemble: Callable,
    make_model: Callable,
) -> WeightedEnsemble[Model]:
    """A WeightedEnsemble with identifiers"""
    ids = "abc"
    weights = {id: 0.3 for id in ids}
    models = {id: make_model() for id in ids}
    return make_weighted_ensemble(path, weights, models)
