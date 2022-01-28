from typing import TypeVar

from pathlib import Path

from automl_common.backend.accessors.accessor import Accessor
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


# TODO assuming a picklable Model
#   Trying to parametrize the saveing and loading functions would
#   lead to any framework using automl_common to not be picklalbe
#   due to lambda's, unknown functions etc..
class ModelAccessor(Accessor[ModelT]):
    """Access state of a Model with a directory on a filesystem

    Manages a directory:
    /<dir>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / model
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
        return self.dir / "model.pkl"

    @property
    def predictions(self) -> PredictionsStore:
        """The predictions store for this ensemble"""
        return self.predictions_store
