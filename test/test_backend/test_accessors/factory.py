from typing import Callable, Mapping, Optional, TypeVar

from pathlib import Path

import numpy as np
from pytest_cases import fixture

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.ensemble import Ensemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)


@fixture(scope="function")
def make_model_accessor() -> Callable[..., ModelAccessor[MT]]:
    """Factory function to make a model accessor

    accessor = make_model_accessor(dir, model=model, predictions={"train":...})
    """

    def _make(
        dir: Path,
        model: Optional[MT] = None,
        predictions: Optional[Mapping[str, np.ndarray]] = None,
    ) -> ModelAccessor[MT]:
        accessor = ModelAccessor[MT](dir=dir)

        if model is not None:
            accessor.save(model)

        if predictions is not None:
            for id, preds in predictions.items():
                accessor.predictions[id] = preds

        return accessor

    return _make


@fixture(scope="function")
def make_ensemble_accessor() -> Callable[..., EnsembleAccessor[ET]]:
    """Factory function to make an ensemble accessor

    accessor = make_ensemble_accessor(
        dir,
        model_dir,
        ensemble=ensemble,
        predictions={"train": ...}
    )
    """

    def _make(
        dir: Path,
        ensemble: Optional[ET] = None,
        predictions: Optional[Mapping[str, np.ndarray]] = None,
    ) -> EnsembleAccessor[ET]:
        accessor = EnsembleAccessor[ET](dir=dir)

        if ensemble is not None:
            accessor.save(ensemble)

        if predictions is not None:
            for id, preds in predictions.items():
                accessor.predictions[id] = preds

        return accessor

    return _make
