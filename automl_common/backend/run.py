from typing import TypeVar, Generic, Iterable, Tuple
from collections.abc import Mapping

import pickle

from automl_common.backend.context import Context, LocalContext


Model = TypeVar("Model")


class Run(Generic[Model]):
    """Interaface to access a run through

    /<id>
        - model
        - {prefix}_predictions
    """

    def __init__(self, id: str, root: str, context: Context):
        self.id = id
        self.root = root
        self.context = context

    @property
    def dir(self) -> str:
        return self.context.join(self.root, str(self.id))

    @property
    def model_path(self) -> str:
        return self.context.join(self.dir, "model")

    def save(self, model: Model) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

    def model(self) -> Model:
        return pickle.load(self.model_path)

    def predictions_path(self, prefix: str, txt: bool = False) -> str:
        if txt:
            return self.context.join(self.dir, f"{prefix}_predictions.txt")
        else:
            return self.context.join(self.dir, f"{prefix}_predictions.txt")

    def save_predictions(self, predictions: np.ndarray, prefix: str, txt: bool = False) -> None:
        if txt:
            with self.context.open(self.predictions_path(prefix, txt), "wb") as f:
                pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
        else:
            with self.context.open(self.predictions_path(prefix), "w") as f:
                np.savetxt(f, predictions)

    def predictions(self, prefix: str, txt) -> np.ndarray:
        with self.context.open(self.predictions(prefix), "rb") as f:
            predicitons = np.load(f)

        return predictions


class Runs(Mapping):
    """Interaface to the runs directory in the backend

    /<dir>
        /<id>
            - model
            - {prefix}_predictions
        /<id>
            - model
            - {prefix}_predictions
        /...
    """

    def __init__(self, dir: str, context: Context):
        """
        Parameters
        ----------
        dir: str
            The directory of the runs

        context: Context
            The context to access the filesystem through
        """
        self.dir = dir
        self.context = context

    def __getitem__(self, id: str) -> Run:
        """Get a run

        Parameters
        ----------
        id: str
            The id of the run
        """
        run_dir = self.context.join(self.dir, id)
        return Run(id=id, dir=run_dir, context=self.context)

    def __iter__(self) -> Iterable[str]:
        """Iterate over runs

        Returns
        -------
        Iterable[Tuple[str, Run]]
            Key, value pairs of identifiers to Run objects
        """
        return iter(self.context.listdir(self.dir))

    def __contains__(self, id: str) -> bool:
        """Whether a given run is contained in the backend

        Parameters
        ----------
        id: str
            The id of the run to get

        Returns
        -------
        bool
            Whether this run is contained in the backend
        """
        path = self.context.join(self.dir, id)
        return self.context.exists(path)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The amount of runs in the backend
        """
        return len(self.context.listdir(self.dir))
