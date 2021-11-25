from abc import abstractmethod
from typing import TypeVar, Generic, Protocol

import pickle

from automl_common.backend.context import Context, LocalContext


Model = TypeVar("Model")

class Run(Generic[Model]):
    """
    /<root>
        /<id>
            - model
            - <prefix>_predictions
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
