import numpy as np

from automl_common.ensemble_building.abstract_ensemble import AbstractEnsemble

EnsembleID = TypeVar("EnsembleID")

class Ensembles:

    def __init__(self, dir: str, context: Context):
        self.dir = dir
        self.context = context

    @property
    def targets_path(self) -> str:
        return self.context.join(self.dir, "targets.npy")

    def save_targets(self, targets: np.ndarray) -> None:
        with self.context.open(self.targets_path, 'wb') as f:
            np.save(f)

    def targets(self) -> np.ndarray:
        with self.context.open(self.targets_path, 'rb') as f:
            targets = np.load(f, allow_pickle=True)

        return targets

class Ensemble(Generic[EnsembleID]):
    """
    /ensembles
        /<id>
            - ensemble
        /<id>
            - ensemble
    """

    def __init__(self, id: EnsembleID, root: str, context: Context):
        self.id = id
        self.root = root
        self.context = context

    @property
    def dir(self) -> str:
        return self.context.join(self.root, str(id))

    @property
    def ensemble_path(self) -> str:
        return self.context.join(self.dir, "ensemble")

    def save(self, ensemble: AbstractEnsemble) -> None:
        with self.context.open(self.ensemble_path, "wb") as f:
            pickle.dump(ensemble, f)

    def load(self) -> AbstractEnsemble:
        with self.context.open(self.ensemble_path, "rb") as f:
            ensemble = pickle.load(f)

        return cast(AbstractEnsemble, ensemble)

    def list(self) -> List[str]:
        return self.context.listdir(self.root)
