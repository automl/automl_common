from typing import Iterator, TypeVar

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.backend.stores.store import StoreView
from automl_common.ensemble import Ensemble

KT = TypeVar("KT")
ET = TypeVar("ET", bound=Ensemble)  # Ensemble Type


class EnsembleStore(StoreView[KT, EnsembleAccessor[ET]]):
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

    def __getitem__(self, key: KT) -> EnsembleAccessor[ET]:
        return self.load(key)

    def load(self, key: KT) -> EnsembleAccessor[ET]:
        """Load the EnsembleAccessor

        Doesn't actually do any loading but it's used with __getitem__
        in StoreView.

        Parameters
        ----------
        key: KT
            The model identifier

        Returns
        -------
        EnsembleAccessor
            A backendwrapper around an Ensemble
        """
        path = self.path(key)
        return EnsembleAccessor[ET](dir=path)

    def __iter__(self) -> Iterator[KT]:
        return iter(key for key in self.iter_all() if self[key].exists())

    def iter_all(self) -> Iterator[KT]:
        """Iterate over the keys of models that may not but have a folder created exist

        Returns
        -------
        Iterable[KT]
        """
        return super().__iter__()
