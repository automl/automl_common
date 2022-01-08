from typing import TypeVar

import pickle

from automl_common.backend.stores import Store

T = TypeVar("T")


class PickleStore(Store[T]):
    """A collection of picklable object"""

    def save(self, picklable: T, key: str) -> None:
        """Saves a picklable object to the key

        Parameters
        ----------
        picklable: T
            A picklable object

        key: str
            The key to save the object under
        """
        path = self.path(key)
        with self.context.open(path, "wb") as f:
            pickle.dump(picklable, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, key: str) -> T:
        """Load a pickled object

        Parameters
        ----------
        key: str
            The key indetifying an object

        Returns
        -------
        T
            Loads the pickled object
        """
        path = self.path(key)
        with self.context.open(path, "rb") as f:
            return pickle.load(f)
