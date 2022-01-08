from typing import Iterator, TypeVar

from automl_common.backend import Backend, PathLike
from automl_common.backend.accessors import ModelAccessor
from automl_common.backend.stores import StoreView
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

    def __init__(self, dir: PathLike, backend: Backend):
        super().__init__(dir=dir, context=backend.context)

        self.backend = backend
        self.load

    def __iter__(self) -> Iterator[str]:
        return iter(self.context.listdir(self.dir))

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
        return ModelAccessor(dir=path, backend=self.backend)
