from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from automl_common.ensemble.weighted_ensemble import WeightedEnsemble

if TYPE_CHECKING:
    from automl_common.backend import Backend


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
