from typing import Callable, Mapping, Optional, TypeVar

from pathlib import Path

import numpy as np
from pytest_cases import fixture

from automl_common.backend.accessors import EnsembleAccessor, ModelAccessor
from automl_common.ensemble import Ensemble
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


@fixture(scope="function")
def make_model_accessor() -> Callable[..., ModelAccessor]:
    """Factory function to make a model accessor

    accessor = make_model_accessor(dir, model=model, predictions={"train":...})
    """

    def _make(
        dir: Path,
        model: Optional[ModelT] = None,
        predictions: Optional[Mapping[str, np.ndarray]] = None,
    ):
        accessor = ModelAccessor[ModelT](dir=dir)

        if model is not None:
            accessor.save(model)

        if predictions is not None:
            for id, preds in predictions.items():
                accessor.predictions[id] = preds

        return accessor

    return _make


@fixture(scope="function")
def make_ensemble_accessor() -> Callable[..., EnsembleAccessor[ModelT]]:
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
        model_dir: Path,
        ensemble: Optional[Ensemble[ModelT]] = None,
        predictions: Optional[Mapping[str, np.ndarray]] = None,
    ) -> EnsembleAccessor[ModelT]:
        accessor = EnsembleAccessor[ModelT](dir=dir, model_dir=model_dir)

        if ensemble is not None:
            accessor.save(ensemble)

        if predictions is not None:
            for id, preds in predictions.items():
                accessor.predictions[id] = preds

        return accessor

    return _make
