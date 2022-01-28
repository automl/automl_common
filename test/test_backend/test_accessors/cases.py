"""
tags:
    "model" - Gives a ModelAccessor
    "ensemble" - Gives an EnsembleAccessor
    "predictions" - Populated with predictions
    "populated" - Populated with an ensemble/model
"""

from typing import Callable, Mapping, Optional, Tuple, TypeVar

from pathlib import Path

import numpy as np
from pytest_cases import case

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.ensemble import Ensemble
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)

ModelAccessorFactory = Callable[
    [Path, Optional[ModelT], Optional[Mapping[str, np.ndarray]]],
    ModelAccessor[ModelT],
]
EnsembleAccessorFactory = Callable[
    [Path, Path, Optional[Ensemble[ModelT]], Optional[Mapping[str, np.ndarray]]],
    Ensemble[ModelT],
]


def _predictions() -> Mapping[str, np.ndarray]:
    return {id: np.array([1]) for id in ["train", "test", "val"]}


def _dirs(path: Path) -> Tuple[Path, Path]:
    return path / "ensemble", path / "models"


@case(tags=["predictions", "model"])
def case_unpopulated_model_accessor_with_predictions(
    path: Path,
    make_model_accessor: ModelAccessorFactory,
) -> ModelAccessor[ModelT]:
    """A ModelAccessor with no model and predictions stored"""
    return make_model_accessor(path, predictions=_predictions())


@case(tags=["populated", "predictions", "model"])
def case_populated_model_accessor_with_predictions(
    path: Path,
    make_model_accessor: ModelAccessorFactory,
    make_model: Callable[..., Model],
) -> ModelAccessor[ModelT]:
    """A ModelAccessor with a model and predictions"""
    return make_model_accessor(path, model=make_model(), predictions=_predictions())


@case(tags=["model"])
def case_unpopulated_model_accessor(
    path: Path,
    make_model_accessor: ModelAccessorFactory,
) -> ModelAccessor[ModelT]:
    """A ModelAccessor with no model and no predictions"""
    return make_model_accessor(path)


@case(tags=["populated", "model"])
def case_populated_model_accessor(
    path: Path,
    make_model_accessor: ModelAccessorFactory,
    make_model: Callable[..., Model],
) -> ModelAccessor[ModelT]:
    """A ModelAccessor with a model and no predictions"""
    return make_model_accessor(path, model=make_model())


@case(tags=["predictions", "ensemble"])
def case_unpopulated_ensemble_accessor_with_predictions(
    path: Path,
    make_ensemble_accessor: EnsembleAccessorFactory,
) -> EnsembleAccessor[ModelT]:
    """An EnsembleAccessor with no ensemble and with predictions"""
    ensemble_dir, model_dir = _dirs(path)
    return make_ensemble_accessor(
        dir=ensemble_dir, model_dir=model_dir, predictions=_predictions()
    )


@case(tags=["populated", "predictions", "ensemble"])
def case_populated_ensemble_accessor_with_predictions(
    path: Path,
    make_ensemble_accessor: EnsembleAccessorFactory,
    make_ensemble: Callable[..., Ensemble[ModelT]],
    make_model: Callable[..., Model],
) -> EnsembleAccessor[ModelT]:
    """An EnsembleAccessor with an ensemble and with predictions"""
    ensemble_dir, model_dir = _dirs(path)
    ensemble = make_ensemble(model_dir, models={id: make_model() for id in "abc"})
    return make_ensemble_accessor(
        dir=ensemble_dir,
        model_dir=model_dir,
        ensemble=ensemble,
        predictions=_predictions(),
    )


@case(tags=["ensemble"])
def case_unpopulated_ensemble_accessor(
    path: Path,
    make_ensemble_accessor: EnsembleAccessorFactory,
) -> EnsembleAccessor[ModelT]:
    """An EnsembleAccessor with no ensemble and no predictions"""
    ensemble_dir, model_dir = _dirs(path)
    return make_ensemble_accessor(dir=ensemble_dir, model_dir=model_dir)


@case(tags=["populated", "ensemble"])
def case_populated_ensemble_accessor(
    path: Path,
    make_ensemble_accessor: EnsembleAccessorFactory,
    make_ensemble: Callable[..., Ensemble[ModelT]],
    make_model: Callable[..., Model],
) -> EnsembleAccessor[ModelT]:
    """An EnsembleAccessor with an ensemble and no predictions"""
    ensemble_dir, model_dir = _dirs(path)
    ensemble = make_ensemble(model_dir, models={id: make_model() for id in "abc"})
    return make_ensemble_accessor(
        dir=ensemble_dir,
        model_dir=model_dir,
        ensemble=ensemble,
    )
