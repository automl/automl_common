from typing import TypeVar

from pathlib import Path

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.backend.stores.store import StoreView
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class EnsembleStore(StoreView[EnsembleAccessor[ModelT]]):
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

    def __init__(self, dir: Path, model_dir: Path):
        """
        Parameters
        ----------
        dir: Path
            The path to where the ensembles are stored

        model_dir: Path
            The path to where the models are stored
        """
        super().__init__(dir)

        if not model_dir.exists():
            model_dir.mkdir()

        self.model_dir = model_dir

    def __getitem__(self, key: str) -> EnsembleAccessor[ModelT]:
        return self.load(key)

    def load(self, key: str) -> EnsembleAccessor[ModelT]:
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
        return EnsembleAccessor[ModelT](dir=path, model_dir=self.model_dir)
