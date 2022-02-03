from typing import TypeVar

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.backend.stores.store import StoreView
from automl_common.ensemble import Ensemble

ET = TypeVar("ET", bound=Ensemble)  # Ensemble Type


class EnsembleStore(StoreView[EnsembleAccessor[ET]]):
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

    def __getitem__(self, key: str) -> EnsembleAccessor[ET]:
        return self.load(key)

    def load(self, key: str) -> EnsembleAccessor[ET]:
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
        return EnsembleAccessor[ET](dir=path)
