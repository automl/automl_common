from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Mapping, Optional, TypeVar

import pickle
from pathlib import Path

from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.backend.stores.model_store import FilteredModelStore
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model

if TYPE_CHECKING:
    from automl_common.backend import Backend, PathLike


ModelT = TypeVar("ModelT", bound=Model)


class EnsembleAccessor(Mapping[str, ModelAccessor[ModelT]]):
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
        backend: Backend[ModelT],
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

        self._predictions_store = PredictionsStore(dir, self.context)
        self._ids: Optional[List[str]] = None

    @property
    def ids(self) -> List[str]:
        """The list of identifiers associated with this ensemble"""
        if not self._ids:
            ensemble = self.load()
            self._ids = list(ensemble.keys())

        return self._ids

    @property
    def predictions(self) -> PredictionsStore:
        """Access to dictlike view of predicitons saved for this ensemble"""
        return self._predictions_store

    @property
    def models(self) -> FilteredModelStore[ModelT]:
        """Return the models contained in this ensemble

        Returns
        -------
        FilteredModelStore
            A model store filtered by the ids of this ensemble
        """
        return FilteredModelStore(dir=self.dir, backend=self.backend, ids=self.ids)

    @property
    def path(self) -> Path:
        """Path to the ensemble object"""
        return self.dir / "ensemble"

    def exists(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the ensemble exists or not
        """
        return self.path.exists()

    def load(self) -> Ensemble:
        """Load a stored ensemble

        Returns
        -------
        EnsembleT
            The loaded ensemble
        """
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, ensemble: Ensemble) -> None:
        """Save an ensemble

        Parameters
        ----------
        ensemlbe: EnsembleT
            The ensemble object to save
        """
        with open(self.path, "wb") as f:
            pickle.dump(ensemble, f)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.path.exists()

    def __getitem__(self, key: str) -> ModelAccessor[ModelT]:
        return self.models[key]

    def __len__(self) -> int:
        return len(self.ids)
