from typing import Collection, Iterator, Optional, TypeVar

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

    def __init__(self, dir: Path, ids: Optional[Collection[str]] = None):
        """
        Parameters
        ----------
        dir : Path
            The path to the directory

        ids : Optional[Collection[str]] = None
            An optional set of ids to filter by
        """
        if ids is not None and len(ids) == 0:
            raise ValueError("Can't have empty `ids` on a ModelStore")

        super().__init__(dir)
        self.ids = ids

    def __iter__(self) -> Iterator[str]:
        iterator = super().__iter__()
        if self.ids is None:
            return iterator
        else:
            return iter(id for id in iterator if id in self.ids)

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
        if self.ids is not None and key not in self.ids:
            raise KeyError(f"{key} not in identifiers {self.ids}")

        path = self.path(key)
        return ModelAccessor(dir=path)

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
        if self.ids is not None and key not in self.ids:
            raise ValueError(f"{key} not in identifiers {self.ids}")

        path = self.path(key)
        return ModelAccessor(dir=path)
