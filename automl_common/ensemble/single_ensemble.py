from typing import TypeVar

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)


class SingleEnsemble(WeightedEnsemble[MT]):
    """An ensemble of just a single model"""

    def __init__(self, model_store: ModelStore[MT], model_id: str):
        """
        Parameters
        ----------
        model_store: ModelStore[MT]
            The path to the models

        identifier: str
            The identifier of the single model
        """
        if model_id == "":
            raise ValueError(f"Found empty string as identifier for {self.__class__.__name__}")

        super().__init__(
            model_store=model_store,
            weighted_ids={model_id: 1.0},
        )
        self.model_id = model_id

    @property
    def model(self) -> MT:
        """Returns the single model in the ensemble"""
        return self[self.model_id]
