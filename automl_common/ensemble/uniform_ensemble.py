from typing import Sequence, TypeVar

from automl_common.backend import Backend
from automl_common.ensemble import WeightedEnsemble

Model = TypeVar("Model")


class UniformEnsemble(WeightedEnsemble):
    """An ensemble of models, each with equal weight"""

    def __init__(self, backend: Backend, identifiers: Sequence[str]):
        """
        Parameters
        ----------
        backend: Backend
            The backend to use

        identifier: Sequence[str]
            The identifiers of the ensemble
        """
        weight = 1.0 / len(identifiers)
        super().__init__(
            backend=backend,
            weighted_identifiers={id: weight for id in identifiers},
        )
