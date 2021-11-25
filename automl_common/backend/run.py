from typing import Any, TypeVar, Generic, Iterable, Tuple
from collections.abc import Mapping

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

    def __init__(self, id: str, dir: str, context: Context):
        """
        Parameters
        ----------
        id: str
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
        if not self.exists():
            self._setup()

        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

    def model(self) -> Model:
        """Return the model for this run"""
        return pickle.load(self.model_path)

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

        with self.context.open(self.predictions_path(prefix), "w") as f:
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
        with self.context.open(self.predictions(prefix), "rb") as f:
            predicitons = np.load(f)

        return predictions

    def setup(self) -> None:
        if self.exists is True:
            raise RuntimeError(f"Run {self.id} already exists at {self.dir}")

        self.context.mkdir(self.dir)
        self.exists = True


class Runs(Mapping):
    """Interaface to the runs directory in the backend

    /<dir>
        /<id>
            - model
            - {prefix}_predictions
        /<id>
            - model
            - {prefix}_predictions
        /...
    """

    def __init__(self, dir: str, context: Context):
        """
        Parameters
        ----------
        dir: str
            The directory of the runs

        context: Context
            The context to access the filesystem through
        """
        self.dir = dir
        self.context = context

    def __getitem__(self, id: Any) -> Run:
        """Get a run

        Parameters
        ----------
        id: Any
            The id of the run
        """
        run_dir = self.context.join(self.dir, str(id))
        return Run(id=str(id), dir=run_dir, context=self.context)

    def __iter__(self) -> Iterable[str]:
        """Iterate over runs

        Returns
        -------
        Iterable[Tuple[str, Run]]
            Key, value pairs of identifiers to Run objects
        """
        return iter(self.context.listdir(self.dir))

    def __contains__(self, id: Any) -> bool:
        """Whether a given run is contained in the backend

        Parameters
        ----------
        id: Any
            The id of the run to get

        Returns
        -------
        bool
            Whether this run is contained in the backend
        """
        path = self.context.join(self.dir, str(id))
        return self.context.exists(path)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The amount of runs in the backend
        """
        return len(self.context.listdir(self.dir))
