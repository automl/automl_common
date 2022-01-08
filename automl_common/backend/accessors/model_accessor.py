from typing import Generic, TypeVar

import pickle
from pathlib import Path

from automl_common.backend import Backend, PathLike
from automl_common.backend.stores import PredictionsStore

Model = TypeVar("Model")


# TODO assuming a picklable Model
#   Trying to parametrize the saveing and loading functions would
#   lead to any framework using automl_common to not be picklalbe.
class ModelAccessor(Generic[Model]):
    """Access state of a Model with a directory on a filesystem

    Manages a directory:
    /<dir>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / model
        / ...
    """

    def __init__(
        self,
        dir: PathLike,
        backend: Backend,
    ):
        """
        Parameters
        ----------
        dir: PathLike
            The directory to load and store from

        context: Context
            A context object to iteract with a filesystem
        """
        self.backend = backend
        self.context = backend.context

        self.dir: Path
        if isinstance(dir, Path):
            self.dir = dir
        else:
            self.dir = self.context.as_path(dir)

        self.predictions_store = PredictionsStore(dir, self.context)

    @property
    def path(self) -> Path:
        """Path to the model object"""
        return self.dir / "model"

    @property
    def predictions(self) -> PredictionsStore:
        """Return the predictions store for this model

        Returns
        -------
        PredictionsStore
            A store of predicitons for the model encapsulated by this ModelBackend

        """
        return self.predictions_store

    def exists(self) -> bool:
        """
        Returns
        -------
        bool
            Whether a saved model exists or not
        """
        return self.context.exists(self.path)

    def load(self) -> Model:
        """Get the model in this model store

        Returns
        -------
        Model
            The loaded model
        """
        with self.context.open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, model: Model) -> None:
        """Save the model

        Parameters
        ----------
        model: Model
            The model to save
        """
        with self.context.open(self.path, "wb") as f:
            pickle.dump(model, f)
