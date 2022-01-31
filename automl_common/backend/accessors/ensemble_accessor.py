from typing import Iterator, List, Mapping, Optional, TypeVar

from pathlib import Path

from automl_common.backend.accessors.accessor import Accessor
from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.backend.stores.model_store import FilteredModelStore
from automl_common.backend.stores.predictions_store import PredictionsStore
from automl_common.ensemble.ensemble import Ensemble
from automl_common.model import Model

MT = TypeVar("MT", bound=Model)
ET = TypeVar("ET", bound=Ensemble)  # Ensemble Type


class EnsembleAccessor(Accessor[ET], Mapping[str, ModelAccessor[MT]]):
    """A wrapper to help with accessing an ensemble and it's models on filesystem

    Manages a directory:
    /<path>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / ensemble
        / ...
    """

    def __init__(
        self,
        dir: Path,
        model_dir: Path,
    ):
        """
        Parameters
        ----------
        dir: Path
            The directory to load and store from
        """
        super().__init__(dir)
        self.model_dir = model_dir
        self.predictions_store = PredictionsStore(self.dir)

        # Cached location for ids in the ensemble
        self._ids: Optional[List[str]] = None

    @property
    def ids(self) -> List[str]:
        """The model ids of models in this ensemble"""
        if not self.exists():
            return []

        return self.load().ids

    @property
    def models(self) -> FilteredModelStore[MT]:
        """Return the models contained in this ensemble

        Returns
        -------
        FilteredModelStore
            A model store filtered by the ids of this ensemble
        """
        if not self.exists():
            raise RuntimeError(f"No saved ensemble found at {self.path}")

        return FilteredModelStore[MT](dir=self.model_dir, ids=self.ids)

    @property
    def path(self) -> Path:
        """Path to the ensemble object"""
        return self.dir / "ensemble.pkl"

    @property
    def predictions(self) -> PredictionsStore:
        """The predictions store for this ensemble"""
        return self.predictions_store

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.ids

    def __getitem__(self, key: str) -> ModelAccessor[MT]:
        if not self.exists():
            raise KeyError(key)

        return self.models[key]

    def __len__(self) -> int:
        return len(self.ids)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EnsembleAccessor)
            and self.dir == other.dir
            and self.model_dir == other.model_dir
            and self.ids == other.ids
        )
