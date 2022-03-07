from typing import Collection, TypeVar

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)
ID = TypeVar("ID")


class UniformEnsemble(WeightedEnsemble[ID, MT]):
    """An ensemble of models, each with equal weight"""

    def __init__(self, model_store: ModelStore[ID, MT], ids: Collection[ID]):
        """
        Parameters
        ----------
        model_store: ModelStore[ID, MT]
            The path to the models

        ids: Collection[ID]
            The ids of the ensemble
        """
        if len(ids) == 0:
            raise ValueError("Instantiated ensemble with empty `ids`")

        weight = 1.0 / len(ids)
        super().__init__(
            model_store=model_store,
            weighted_ids={id: weight for id in ids},
        )
