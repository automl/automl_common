"""
tags:
    "params" - If it gives params for constructing an ensemble
    {"mock", "single", "uniform", "weighted"} - type of ensemble
"""
from typing import Callable, List, Mapping, Type, TypeVar, Union

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
from test.test_model.mocks import MockModel

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)


@case(tags=["mock"])
@parametrize("n_models", [1, 10])
@parametrize("model_type", [MockModel])
def case_mock(
    path: Path,
    n_models: int,
    model_type: Type[MT],
    make_ensemble: Callable[..., MockEnsemble[MT]],
) -> MockEnsemble[MT]:
    """A MockEnsemble with identifiers"""
    return make_ensemble(models=n_models, model_type=model_type, path=path)


@case(tags=["single"])
@parametrize("model_type", [MockModel])
def case_single(
    path: Path,
    model_type: Type[MT],
    make_single_ensemble: Callable[..., SingleEnsemble[MT]],
) -> SingleEnsemble[MT]:
    """A SingleEnsemble with identifier"""
    return make_single_ensemble(model_type=model_type, path=path)


@case(tags=["uniform"])
@parametrize("n_models", [1, 3, 10])
@parametrize("model_type", [MockModel])
def case_uniform(
    path: Path,
    n_models: int,
    model_type: Type[MT],
    make_uniform_ensemble: Callable[..., UniformEnsemble[MT]],
) -> UniformEnsemble[MT]:
    """A UniformEnsemble with identifiers"""
    return make_uniform_ensemble(models=n_models, model_type=model_type, path=path)


@case(tags=["weighted"])
@parametrize(
    "n_models, weights",
    [
        (1, [1.0]),
        (3, {"0": 3.4, "1": -2.1, "2": 14.0}),
        (10, [0.1 for _ in range(10)]),
    ],
)
@parametrize("model_type", [MockModel])
def case_weighted(
    path: Path,
    n_models: int,
    model_type: Type[MT],
    weights: Union[List[float], Mapping[str, float]],
    make_weighted_ensemble: Callable[..., WeightedEnsemble[MT]],
) -> WeightedEnsemble[MT]:
    """A WeightedEnsemble with identifiers"""
    return make_weighted_ensemble(
        models=n_models,
        model_type=model_type,
        weights=weights,
        path=path,
    )
