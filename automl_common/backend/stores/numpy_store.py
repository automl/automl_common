import numpy as np

from automl_common.backend.stores.store import Store


class NumpyStore(Store[np.ndarray]):
    """A collection of numpy objects"""

    def save(self, array: np.ndarray, key: str) -> None:
        """Save a numpy array as key

        Parameters
        ----------
        array: np.ndarray
            The array to save

        key: str
            The key to save it as
        """
        path = self.path(key)
        with self.context.open(path, "wb") as f:
            np.save(f, array)

    def load(self, key: str) -> np.ndarray:
        """Load a numpy array identified by key

        Parameters
        ----------
        key: str
            The key identifying the array

        Returns
        -------
        Optional[np.ndarray]
            The loaded array
        """
        path = self.path(key)
        if not path.exists():
            raise KeyError(key)

        with self.context.open(path, "rb") as f:
            return np.load(f)
