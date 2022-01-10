from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

if TYPE_CHECKING:
    from automl_common.backend import Backend

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
