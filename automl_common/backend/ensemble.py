from typing import Any, cast

import pickle

from ..ensemble_building.abstract_ensemble import AbstractEnsemble
from .context import Context


class Ensemble:
    """Interface to a specific ensemble saved in the backend

    /<id>
        - ensemble
    """

    def __init__(self, id: Any, dir: str, context: Context):
        """
        Parameters
        ----------
        id: str
            A unique identifier for an ensemble

        dir: str
            The directory where this ensemble is stored

        context: Context
            The context to access the filesystem with
        """
        self.id = id
        self.dir = dir
        self.context = context

        # Created whenever we save an ensemble
        self.exists = self.context.exists(self.dir)

    @property
    def ensemble_path(self) -> str:
        """The path to the ensemle model"""
        return self.context.join(self.dir, "ensemble")

    def save(self, ensemble: AbstractEnsemble) -> None:
        """Save an ensemble to a file

        Parameters
        ----------
        ensemble: AbstractEnsemble
            The ensemble object to save
        """
        if not self.exists:
            self.setup()

        with self.context.open(self.ensemble_path, "wb") as f:
            pickle.dump(ensemble, f)

    def load(self) -> AbstractEnsemble:
        """Save an ensemble to the filesystem"""
        if not self.exists:
            raise RuntimeError(
                f"Ensemble with id {self.id} at {self.dir} has no ensemble saved"
            )

        with self.context.open(self.ensemble_path, "rb") as f:
            ensemble = pickle.load(f)

        return cast(AbstractEnsemble, ensemble)

    def setup(self) -> None:
        if self.exists is True:
            raise RuntimeError(f"Ensemble {self.id} already exists at {self.dir}")

        self.context.mkdir(self.dir)
        self.exists = True

    def __eq__(self, other: Any) -> bool:
        """Two ensembles are equal if they have the same id and the same dir

        Parameters
        ----------
        other: Any
            The object to compare to

        Returns
        -------
        bool
            Whether this is equal to other
        """
        if not isinstance(other, Ensemble):
            raise NotImplementedError()

        return str(self.id) == str(other.id) and self.dir == other.dir
