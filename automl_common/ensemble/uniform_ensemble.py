from typing import Sequence, TypeVar

from pathlib import Path

from automl_common.ensemble.weighted_ensemble import WeightedEnsemble
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class UniformEnsemble(WeightedEnsemble[ModelT]):
    """An ensemble of models, each with equal weight"""

    def __init__(self, model_dir: Path, identifiers: Sequence[str]):
        """
        Parameters
        ----------
        model_dir: Path
            The path to the models

        identifier: Sequence[str]
            The identifiers of the ensemble
        """
        if len(identifiers) == 0:
            raise ValueError("Instantiated ensemble with empty `identifiers`")

        weight = 1.0 / len(identifiers)
        super().__init__(
            model_dir=model_dir,
            weighted_identifiers={id: weight for id in identifiers},
        )
