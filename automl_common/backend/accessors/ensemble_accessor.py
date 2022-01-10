from __future__ import annotations

from typing import TYPE_CHECKING

import pickle
from pathlib import Path

from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.ensemble.ensemble import Ensemble

if TYPE_CHECKING:
    from automl_common.backend import Backend, PathLike


# TODO make iterable
class EnsembleAccessor:
    """The state of an Ensemble with a directory on a filesystem.

    As automl_common manages ensembling in general, we can keep
    these as picklable items and hence implement what it means
    to load and save an ensemble.

    Manages a directory:
    /<path>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / ensemble
        / ...
    Also uses
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
        self.context = backend.context
        self.backend = backend

        self.dir: Path
        if isinstance(dir, Path):
            self.dir = dir
        else:
            self.dir = self.context.as_path(dir)

        self.predictions_store = PredictionsStore(dir, self.context)

    @property
    def predictions(self) -> PredictionsStore:
        """Access to dictlike view of predicitons saved for this ensemble"""
        return self.predictions_store

    @property
    def path(self) -> Path:
        """Path to the ensemble object"""
        return self.path / "ensemble.json"

    def exists(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the ensemble exists or not
        """
        return self.context.exists(self.path)

    def load(self) -> Ensemble:
        """Load a stored ensemble

        Returns
        -------
        EnsembleT
            The loaded ensemble
        """
        # TODO, load in the models too
        with open(self.path, "rb") as f:
            ensemble = pickle.load(f)

            # We need to inject the current backend we're using
            ensemble.backend = self.backend

            return ensemble

    def save(self, ensemble: Ensemble) -> None:
        """Save an ensemble

        Parameters
        ----------
        ensemlbe: EnsembleT
            The ensemble object to save
        """
        with open(self.path, "wb") as f:
            pickle.dump(ensemble, f)
