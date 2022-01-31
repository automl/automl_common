from typing import TypeVar

from pathlib import Path

from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)


class SingleEnsemble(WeightedEnsemble[MT]):
    """An ensemble of just a single model"""

    def __init__(self, model_dir: Path, model_id: str):
        """
        Parameters
        ----------
        model_dir: Path
            The path to the models

        identifier: str
            The identifier of the single model
        """
        if model_id == "":
            raise ValueError("Found empty string as identifier for SingleEnsemble")

        super().__init__(
            model_dir=model_dir,
            weighted_ids={model_id: 1.0},
        )
        self.model_id = model_id

    @property
    def model(self) -> MT:
        """Returns the single model in the ensemble"""
        return self[self.model_id]
