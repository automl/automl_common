from typing import Iterator

from automl_common.backend import Backend, PathLike
from automl_common.backend.accessors import EnsembleAccessor
from automl_common.backend.stores import StoreView


class EnsembleStore(StoreView[EnsembleAccessor]):
    """A store of linking keys to EnsembleAccessor

    Manages a directory:
    /<path>
        /<ensemble_key1>
            / predictions_{}.npy
            / ensemble
            / ...
        /<ensemble_key2>
            / predictions_{}.npy
            / ensemble
            / ...
        / ...
    """

    def __init__(self, dir: PathLike, backend: Backend):
        """
        Parameters
        ----------
        dir: PathLike
            The path to where the ensembles are stored

        backend: Backend
            The backend to use. This is not a simple Context as the EnsembleStore
            must be able to access Models directly with an id.
        """
        super().__init__(dir, backend.context)
        self.backend = backend

    def __iter__(self) -> Iterator[str]:
        return iter(self.context.listdir(self.dir))

    def load(self, key: str) -> EnsembleAccessor:
        """Load the EnsembleAccessor

        Doesn't actually do any loading but it's used with __getitem__
        in StoreView.

        Parameters
        ----------
        key: str
            The model identifier

        Returns
        -------
        EnsembleAccessor
            A backendwrapper around an Ensemble
        """
        path = self.path(key)
        return EnsembleAccessor(dir=path, backend=self.backend)
