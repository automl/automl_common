from typing import Iterator, TypeVar

import re
from pathlib import Path

import numpy as np

from automl_common.backend.stores.store import Store

KT = TypeVar("KT")


class NumpyStore(Store[KT, np.ndarray]):
    """A collection of numpy objects"""

    pattern = r"(.*).npy"

    def path(self, key: KT) -> Path:
        """Get path for the numpy object

        Parameters
        ----------
        key: KT
            The object saved under KT

        Returns
        -------
        Path
            The path to the numpy array
        """
        return self.dir / f"{self.__key_to_str__(key)}.npy"

    def save(self, array: np.ndarray, key: KT) -> None:
        """Save a numpy array as key

        Parameters
        ----------
        array: np.ndarray
            The array to save

        key: KT
            The key to save it as
        """
        with self.path(key).open("wb") as f:
            np.save(f, array)

    def load(self, key: KT) -> np.ndarray:
        """Load a numpy array identified by key

        Parameters
        ----------
        key: KT
            The key identifying the array

        Returns
        -------
        Optional[np.ndarray]
            The loaded array
        """
        with self.path(key).open("rb") as f:
            return np.load(f)

    def __iter__(self) -> Iterator[KT]:
        matches = iter(re.match(self.pattern, file.name) for file in self.dir.iterdir())
        return iter(self.__name_to_key__(match.group(1)) for match in matches if match is not None)
