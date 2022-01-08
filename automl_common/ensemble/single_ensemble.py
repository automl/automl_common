from typing import TypeVar

from automl_common.backend import Backend
from automl_common.ensemble import WeightedEnsemble
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class SingleEnsemble(WeightedEnsemble[ModelT]):
    """An ensemble of just a single model"""

    def __init__(self, backend: Backend, identifier: str):
        """
        Parameters
        ----------
        backend: Backend
            The backend to use

        identifier: str
            The identifier of the single model
        """
        super().__init__(
            backend=backend,
            weighted_identifiers={identifier: 1.0},
        )
