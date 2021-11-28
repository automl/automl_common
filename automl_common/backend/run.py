from typing import Any, TypeVar, Generic, Iterable, Tuple, cast

import numpy as np
import pickle

from automl_common.backend.context import Context, LocalContext


Model = TypeVar("Model")


class Run(Generic[Model]):
    """Interaface to access a run through.

    /<id>
        - model
        - {prefix}_predictions
    """

    def __init__(self, id: Any, dir: str, context: Context):
        """
        Parameters
        ----------
        id: Any
            The id of this run

        dir: str
            The directory where this run is based

        context: Context
            The context object to access the filesystem with
        """
        self.id = id
        self.dir = dir
        self.context = context

        # We cache whether this run exists when it's created
        # It's created whenever we save a model or predictions
        self.exists = self.context.exists(self.dir)

    @property
    def model_path(self) -> str:
        """The path to the model of this run"""
        return self.context.join(self.dir, "model")

    def has_model(self) -> bool:
        """Whether this run has a model saved"""
        return self.context.exists(self.model_path)

    def save_model(self, model: Model) -> None:
        """Save a model to this run directory

        Parameters
        ----------
        model: Model
            A model object to save, must be picklable
        """
        if not self.exists:
            self.setup()

        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

    def model(self) -> Model:
        """Return the model for this run"""
        with open(self.model_path, "rb") as f:
            return cast(Model, pickle.load(f))

    def predictions_path(self, prefix: str) -> str:
        """Get the path for predictions starting with `prefix`

        Parameters
        ----------
        prefix: str
            The prefix to give the prediciton files

        Returns
        -------
        str
            The predictions filepath /x/y/runs/<id>/{prefix}_predictions.npy
        """
        return self.context.join(self.dir, f"{prefix}_predictions.npy")

    def has_predictions(self, prefix: str) -> bool:
        """Whether predictions starting with `prefix` exist

        Parameters
        ----------
        prefix: str
            The prefix to give the predictions

        Returns
        -------
        bool
            Whether they exist or not
        """
        return self.context.exists(self.predictions_path(prefix))

    def save_predictions(self, predictions: np.ndarray, prefix: str) -> None:
        """Save predictions with a given prefix {prefix}_predictions.npy

        Parameters
        ----------
        predictions: np.ndarray
            The predictions to store

        prefix: str
            The prefix to give the predictions
        """
        if not self.exists:
            self.setup()

        with self.context.open(self.predictions_path(prefix), "wb") as f:
            np.save(f, predictions)

    def predictions(self, prefix: str) -> np.ndarray:
        """Get the predictions with a given prefix

        Parameters
        ----------
        prefix: str
            The prefix of the predictions

        Returns
        -------
        np.ndarray
            The predictions as an np.ndarray
        """
        with self.context.open(self.predictions_path(prefix), "rb") as f:
            predictions = np.load(f)

        return predictions

    def setup(self) -> None:
        if self.exists is True:
            raise RuntimeError(f"Run {self.id} already exists at {self.dir}")

        self.context.mkdir(self.dir)
        self.exists = True

    def __eq__(self, other: Any) -> bool:
        """Two runs are equal if they have the same id and the same dir

        Parameters
        ----------
        other: Any
            The object to compare to

        Returns
        -------
        bool
            Whether this is equal to other
        """
        if not isinstance(other, Run):
            raise NotImplementedError()

        return str(self.id) == str(other.id) and self.dir == other.dir
