from typing import Iterator

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
        return EnsembleAccessor(dir=path, context=self.context)
