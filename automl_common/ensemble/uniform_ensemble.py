from typing import Collection, TypeVar

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)


class UniformEnsemble(WeightedEnsemble[MT]):
    """An ensemble of models, each with equal weight"""

    def __init__(self, model_store: ModelStore[MT], ids: Collection[str]):
        """
        Parameters
        ----------
        model_store: ModelStore[MT]
            The path to the models

        ids: Sequence[str]
            The ids of the ensemble
        """
        if len(ids) == 0:
            raise ValueError("Instantiated ensemble with empty `ids`")

        weight = 1.0 / len(ids)
        super().__init__(
            model_store=model_store,
            weighted_ids={id: weight for id in ids},
        )
