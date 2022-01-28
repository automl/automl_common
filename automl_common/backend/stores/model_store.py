from typing import Collection, Iterator, TypeVar

from pathlib import Path

from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.backend.stores.store import StoreView
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class ModelStore(StoreView[ModelAccessor[ModelT]]):
    """A store of linking keys to ModelAccessor

    Manages a directory:
    /<path>
        /<model_key1>
            / predictions_{}.npy
            / model
            / ...
        /<model_key2>
            / predictions_{}.npy
            / model
            / ...
        / ...
    """

    def __getitem__(self, key: str) -> ModelAccessor[ModelT]:
        """Gets the ModelAccessor for the model associated with a model

        Parameters
        ----------
        key: str
            The model identifier

        Returns
        -------
        ModelAccessor[Model]
            A wrapper around a model in a directory
        """
        return self.load(key)

    def load(self, key: str) -> ModelAccessor[ModelT]:
        """Gets the ModelAccessor for the model associated with a model

        Doesn't actually do any loading but it's used with __getitem__
        in StoreView.

        Parameters
        ----------
        key: str
            The model identifier

        Returns
        -------
        ModelAccessor[Model]
            A wrapper around a model in a directory
        """
        path = self.path(key)
        return ModelAccessor(dir=path)


class FilteredModelStore(ModelStore[ModelT]):
    """A ModelStore additionally filtered out by identifiers"""

    def __init__(self, dir: Path, ids: Collection[str]):
        """
        Parameters
        ----------
        dir: Path
            The directory to check in

        ids: Collection[str]
            The ids to filter by
        """
        if len(ids) == 0:
            raise ValueError("Can't have FilteredModelStore with no ids")

        self.ids = ids
        super().__init__(dir=dir)

    def __iter__(self) -> Iterator[str]:
        existing = {path.name for path in self.dir.iterdir()}
        return iter(id for id in self.ids if id in existing)

    def __getitem__(self, key: str) -> ModelAccessor[ModelT]:
        if key not in self.ids:
            raise KeyError(f"{key} not in identifiers {self.ids}")

        return super().__getitem__(key)

    def load(self, key: str) -> ModelAccessor[ModelT]:
        """Gets the ModelAccessor for the model associated with a model

        Doesn't actually do any loading but it's used with __getitem__
        in StoreView.

        Parameters
        ----------
        key: str
            The model identifier

        Returns
        -------
        ModelAccessor[Model]
            A wrapper around a model in a directory
        """
        if key not in self.ids:
            raise ValueError(f"{key} not in identifiers {self.ids}")

        return super().load(key)
