from typing import TypeVar

from pathlib import Path

from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class SingleEnsemble(WeightedEnsemble[ModelT]):
    """An ensemble of just a single model"""

    def __init__(self, model_dir: Path, identifier: str):
        """
        Parameters
        ----------
        model_dir: Path
            The path to the models

        identifier: str
            The identifier of the single model
        """
        if identifier == "":
            raise ValueError("Found empty string as identifier for SingleEnsemble")

        super().__init__(
            model_dir=model_dir,
            weighted_identifiers={identifier: 1.0},
        )

    @property
    def model(self) -> ModelAccessor[ModelT]:
        """Returns the single model in the ensemble"""
        return next(iter(self.models.values()))
