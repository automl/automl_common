import pickle

from automl_common.ensemble_building.abstract_ensemble import AbstractEnsemble
from automl_common.backend.context import Context


class Ensemble:
    """Interface to a specific ensemble saved in the backend

    /<id>
        - ensemble
    """

    def __init__(self, id: str, dir: str, context: Context):
        """
        Parameters
        ----------
        id: str
            A unique identifier for an ensemble

        dir: str
            The directory where this ensemble is stored

        context: Context
            The context to access the filesystem with
        """
        self.id = id
        self.dir = dir
        self.context = context

        self.exists = False

    @property
    def ensemble_path(self) -> str:
        """The path to the ensemle model"""
        return self.context.join(self.dir, "ensemble")

    def save(self, ensemble: AbstractEnsemble) -> None:
        """Save an ensemble to a file

        Parameters
        ----------
        ensemble: AbstractEnsemble
            The ensemble object to save
        """
        if not self.exists:
            self.setup()

        with self.context.open(self.ensemble_path, "wb") as f:
            pickle.dump(ensemble, f)

    def load(self) -> AbstractEnsemble:
        """Save an ensemble to the filesystem"""
        with self.context.open(self.ensemble_path, "rb") as f:
            ensemble = pickle.load(f)

        return cast(AbstractEnsemble, ensemble)

    def setup(self) -> None:
        if self.exists is True:
            raise RuntimeError(f"Ensemble {self.id} already exists at {self.dir}")

        self.context.mkdir(self.dir)
        self.exists = True
