from typing import TypeVar

from pathlib import Path

from automl_common.backend.accessors.accessor import Accessor
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.ensemble.ensemble import Ensemble

ET = TypeVar("ET", bound=Ensemble)  # Ensemble Type


class EnsembleAccessor(Accessor[ET]):
    """A wrapper to help with accessing an ensemble on filesystem

    Manages a directory:
    /<path>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / ensemble
        / ...
    """

    def __init__(self, dir: Path):
        """
        Parameters
        ----------
        dir: Path
            The directory to load and store from
        """
        super().__init__(dir)
        self.predictions_store = PredictionsStore(self.dir)

    @property
    def path(self) -> Path:
        """Path to the model object"""
        return self.dir / "ensemble.pkl"

    @property
    def predictions(self) -> PredictionsStore:
        """The predictions store for this ensemble"""
        return self.predictions_store
