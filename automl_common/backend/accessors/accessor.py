from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pickle
from pathlib import Path

from automl_common.backend.util.path import rmtree
from automl_common.util.types import EqualityMixin

T = TypeVar("T")


# TODO assuming a picklable object
#   Trying to parametrize the saveing and loading functions would
class Accessor(ABC, EqualityMixin, Generic[T]):
    """Access state of some folder object wit predictions and an object

    Manages a directory:
    /<dir>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / obj
        / ...
    """

    def __init__(self, dir: Path):
        """
        Parameters
        ----------
        dir: Path
            The directory to load and store from
        """
        self.dir = dir

    @property
    @abstractmethod
    def path(self) -> Path:
        """Path to the pickled object"""
        ...

    def exists(self) -> bool:
        """
        Returns
        -------
        bool
            Whether a saved obj exists or not
        """
        return self.path.exists()

    def load(self) -> T:
        """Get the obj in this model store

        Returns
        -------
        T
            The loaded obj
        """
        with self.path.open("rb") as f:
            return pickle.load(f)

    def save(self, obj: T) -> None:
        """Save the obj

        Parameters
        ----------
        model: Model
            The model to save
        """
        with self.path.open("wb") as f:
            pickle.dump(obj, f)

    def delete(self, folder: bool = False) -> None:
        """Delete this obj, optionally deleting its full folder"""
        if folder:
            rmtree(self.dir)
        else:
            self.path.unlink()

    def __str__(self) -> str:
        return f"Accessor: {self.dir}"
