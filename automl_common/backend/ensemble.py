from typing import TypeVar, Generic, Iterable, Tuple
from collections.abc import Mapping

import numpy as np

from automl_common.ensemble_building.abstract_ensemble import AbstractEnsemble


class Ensemble:
    """Interface to a specific ensemble saved in the backend

    /<id>
        - ensemble
    """

    def __init__(self, id: str, dir: str, context: Context):
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

        self.exists = False

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
            self._setup()

        with self.context.open(self.ensemble_path, "wb") as f:
            pickle.dump(ensemble, f)

    def load(self) -> AbstractEnsemble:
        """Save an ensemble to the filesystem"""
        with self.context.open(self.ensemble_path, "rb") as f:
            ensemble = pickle.load(f)

        return cast(AbstractEnsemble, ensemble)

    def _setup(self) -> None:
        if self.exists is True:
            raise RuntimeError(f"Ensemble {self.id} already exists at {self.dir}")

        self.context.mkdir(self.dir)
        self.exists = True


class Ensembles(Mapping):
    """Interface to the ensembles part of the backend

    /ensembles
        /<id>
            - ensemble
        /<id>
            - ensemble
    """

    def __init__(self, dir: str, context: Context):
        """
        Parameters
        ----------
        dir: str
            Where the ensembles part of the backend is located

        context: Context
            The context used for file operations
        """
        self.dir = dir
        self.context = context

    @property
    def targets_path(self) -> str:
        """The path to the targets used for evaluating an ensemble"""
        return self.context.join(self.dir, "targets.npy")

    def save_targets(self, targets: np.ndarray) -> None:
        """Save a set of targets

        Parameters
        ----------
        targets: np.ndarray
            The targets to save
        """
        with self.context.open(self.targets_path, "wb") as f:
            np.save(f)

    def targets(self) -> np.ndarray:
        """Load the targets from disk

        Returns
        -------
        np.ndarray
            The targets used to evaluate an ensemble
        """
        with self.context.open(self.targets_path, "rb") as f:
            targets = np.load(f, allow_pickle=True)

        return targets

    def __getitem__(self, id: Any) -> Ensemble:
        """Get an ensemble

        Parameters
        ----------
        id: str
            The id of the ensemble
        """
        ensemble_dir = self.context.join(self.dir, str(id))
        return Ensemble(id=str(id), dir=ensemble_dir, context=self.context)

    def __iter__(self) -> Iterable[str]:
        """Iterate over ensembles

        Returns
        -------
        Iterable[Tuple[str, Ensemble]]
            Key, value pairs of identifiers to Ensemble objects
        """
        return iter(self.context.listdir(self.dir))

    def __contains__(self, id: Any) -> bool:
        """Whether a given ensemble is contained in the backend

        Parameters
        ----------
        id: str
            The id of the ensemble to get

        Returns
        -------
        bool
            Whether this ensemble is contained in the backend
        """
        path = self.context.join(self.dir, str(id))
        return self.context.exists(path)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The amount of ensembles in the backend
        """
        return len(self.context.listdir(self.dir))
