"""
tags:
    "params" - If it gives params for constructing an ensemble
    "invalid"/"valid" - If the ensemble is in an invalid/valid state
    <type>+ - mock, uniform, weighted, single
"""
from typing import Any, Callable, Collection, Dict, Tuple, Type

from pathlib import Path

from pytest_cases import case, parametrize

from automl_common.ensemble import (
    Ensemble,
    SingleEnsemble,
    UniformEnsemble,
    WeightedEnsemble,
)
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble


@case(tags=["mock", "uniform", "weighted", "params", "invalid"])
@parametrize("ids", [[], {}, ()])
@parametrize("cls", [MockEnsemble, UniformEnsemble, WeightedEnsemble])
def case_params_empty_identifiers(
    path: Path,
    ids: Collection[str],
    cls: Type[Ensemble],
) -> Tuple[Type[Ensemble], Dict[str, Any], Type[Exception], str]:
    """Should not instantiate with empty identifiers container

    This will cause iteration to fail and generally not a good idea.
    """
    msg = "Instantiated ensemble with empty `identifiers`"
    args = {
        "model_dir": path,
        "identifiers": ids,
    }

    if cls == WeightedEnsemble:
        del args["identifiers"]
        args["weighted_identifiers"] = {id: 0.0 for id in ids}

    err = ValueError
    return cls, args, err, msg


@case(tags=["mock", "uniform", "weighted", "params", "invalid"])
@parametrize("ids", [["a", "b", "c"]])
@parametrize("cls", [MockEnsemble, UniformEnsemble, WeightedEnsemble])
def case_params_identifiers_no_models(
    path: Path,
    ids: Collection[str],
    cls: Type[Ensemble],
) -> Tuple[Type[Ensemble], Dict[str, Any], Type[Exception], str]:
    """Should not instantiate with identifiers but no models stored

    To construct the Ensemble object in a valid state, we assume it
    must contain model objects to work with.
    """
    msg = "No model for id"
    args = {
        "model_dir": path,
        "identifiers": ids,
    }

    if cls == WeightedEnsemble:
        del args["identifiers"]
        args["weighted_identifiers"] = {id: 0.3 for id in ids}

    err = ValueError
    return cls, args, err, msg


@case(tags=["single", "params", "invalid"])
@parametrize("id", [""])
def case_params_single_ensemble_with_empty_string(
    path: Path,
    id: str,
) -> Tuple[Type[Ensemble], Dict[str, Any], Type[Exception], str]:
    """Should not instantiate with empty string id

    The empty string will append to a path but will not act as a key
        `path / "" == path`
    """
    msg = "Found empty string as identifier for SingleEnsemble"
    args = {
        "model_dir": path,
        "identifier": id,
    }
    err = ValueError
    return SingleEnsemble[Model], args, err, msg


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
