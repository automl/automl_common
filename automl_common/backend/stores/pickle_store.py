from typing import Iterator, TypeVar

import pickle
import re
from pathlib import Path

from automl_common.backend.stores.store import Store

T = TypeVar("T")


class PickleStore(Store[T]):
    """A collection of picklable object"""

    pattern = r"(.*)\.pkl"

    def path(self, key: str) -> Path:
        """The path to the pickled item

        Parameters
        ----------
        key: str
            The key of the item

        Returns
        -------
        Path
            The path to the item
        """
        return self.dir / f"{key}.pkl"

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

    def __iter__(self) -> Iterator[str]:
        files = self.context.listdir(self.dir)
        matches = iter(re.match(self.pattern, file) for file in files)
        return iter(match.group(1) for match in matches if match is not None)
