from typing import Iterator, TypeVar

from automl_common.backend.accessors.model_accessor import ModelAccessor
from automl_common.backend.stores.store import StoreView
from automl_common.model import Model

KT = TypeVar("KT")
MT = TypeVar("MT", bound=Model)


class ModelStore(StoreView[KT, ModelAccessor[MT]]):
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

    def __getitem__(self, key: KT) -> ModelAccessor[MT]:
        """Gets the ModelAccessor for the model associated with a model

        Overwrites default behaviour of giving KeyError if doesn't exist.

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

    def load(self, key: KT) -> ModelAccessor[MT]:
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

    def __iter__(self) -> Iterator[KT]:
        return iter(key for key in self.iter_all() if self[key].exists())

    def iter_all(self) -> Iterator[KT]:
        """Iterate over the keys of models that may not but have a folder created exist

        Returns
        -------
        Iterable[KT]
        """
        return super().__iter__()
