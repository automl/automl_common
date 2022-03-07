"""
tags:
    "model" - Gives a ModelAccessor
    "ensemble" - Gives an EnsembleAccessor
    "predictions" - Populated with predictions
    "populated" - Populated with an ensemble/model
"""

from typing import Callable, Mapping, Tuple, Type, TypeVar, Union

from pathlib import Path

from pytest_cases import case, parametrize

import numpy as np

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.ensemble import Ensemble
from automl_common.model import Model

from test.test_model.mocks import MockModel

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)
ID = TypeVar("ID")


def _predictions() -> Mapping[str, np.ndarray]:
    return {id: np.array([1]) for id in ["train", "test", "val"]}


def _dirs(path: Path) -> Tuple[Path, Path]:
    return path / "ensemble", path / "models"


@case(tags=["predictions", "model"])
def case_unpopulated_model_accessor_with_predictions(
    path: Path,
    make_model_accessor: Callable[..., ModelAccessor[MT]],
) -> ModelAccessor[MT]:
    """A ModelAccessor with no model and predictions stored"""
    return make_model_accessor(path, predictions=_predictions())


@case(tags=["populated", "predictions", "model"])
def case_populated_model_accessor_with_predictions(
    path: Path,
    make_model_accessor: Callable[..., ModelAccessor[MT]],
    make_model: Callable[..., Model],
) -> ModelAccessor[MT]:
    """A ModelAccessor with a model and predictions"""
    return make_model_accessor(path, model=make_model(), predictions=_predictions())


@case(tags=["model"])
def case_unpopulated_model_accessor(
    path: Path,
    make_model_accessor: Callable[..., ModelAccessor[MT]],
) -> ModelAccessor[MT]:
    """A ModelAccessor with no model and no predictions"""
    return make_model_accessor(path)


@case(tags=["populated", "model"])
def case_populated_model_accessor(
    path: Path,
    make_model_accessor: Callable[..., ModelAccessor[MT]],
    make_model: Callable[..., Model],
) -> ModelAccessor[MT]:
    """A ModelAccessor with a model and no predictions"""
    return make_model_accessor(path, model=make_model())


@case(tags=["predictions", "ensemble"])
def case_unpopulated_ensemble_accessor_with_predictions(
    path: Path,
    make_ensemble_accessor: Callable[..., EnsembleAccessor[ET]],
) -> EnsembleAccessor[ET]:
    """An EnsembleAccessor with no ensemble and with predictions"""
    ensemble_dir, model_dir = _dirs(path)
    return make_ensemble_accessor(dir=ensemble_dir, predictions=_predictions())


@case(tags=["populated", "predictions", "ensemble"])
@parametrize("models", [1, 5, ("a", "b", "c")])
@parametrize("model_type", [MockModel])
def case_populated_ensemble_accessor_with_predictions(
    path: Path,
    models: Union[int, Tuple[str]],
    model_type: Type[MT],
    make_ensemble_accessor: Callable[..., EnsembleAccessor[ET]],
    make_ensemble: Callable[..., Ensemble[ID, MT]],
) -> EnsembleAccessor[ET]:
    """An EnsembleAccessor with an ensemble and with predictions"""
    ensemble_dir, model_dir = _dirs(path)
    ensemble = make_ensemble(models=models, path=model_dir, model_type=model_type)
    return make_ensemble_accessor(
        dir=ensemble_dir,
        ensemble=ensemble,
        predictions=_predictions(),
    )


@case(tags=["ensemble"])
def case_unpopulated_ensemble_accessor(
    path: Path,
    make_ensemble_accessor: Callable[..., EnsembleAccessor[ET]],
) -> EnsembleAccessor[ET]:
    """An EnsembleAccessor with no ensemble and no predictions"""
    ensemble_dir, model_dir = _dirs(path)
    return make_ensemble_accessor(dir=ensemble_dir)


@case(tags=["populated", "ensemble"])
@parametrize("models", [1, 5, ("a", "b", "c")])
@parametrize("model_type", [MockModel])
def case_populated_ensemble_accessor(
    path: Path,
    models: Union[int, Tuple[str]],
    model_type: Type[MT],
    make_ensemble_accessor: Callable[..., EnsembleAccessor[ET]],
    make_ensemble: Callable[..., Ensemble[ID, MT]],
) -> EnsembleAccessor[ET]:
    """An EnsembleAccessor with an ensemble and no predictions"""
    ensemble_dir, model_dir = _dirs(path)
    ensemble = make_ensemble(models=models, path=model_dir, model_type=model_type)
    return make_ensemble_accessor(
        dir=ensemble_dir,
        ensemble=ensemble,
    )
