from typing import Iterator, TypeVar

import pickle
import re
from pathlib import Path

from automl_common.backend.stores.store import Store

VT = TypeVar("VT")
KT = TypeVar("KT")


class PickleStore(Store[KT, VT]):
    """A collection of picklable object"""

    pattern = r"(.*)\.pkl"

    def path(self, key: KT) -> Path:
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
        return self.dir / f"{self.__key_to_str__(key)}.pkl"

    def save(self, picklable: VT, key: KT) -> None:
        """Saves a picklable object to the key

        Parameters
        ----------
        picklable: T
            A picklable object

        key: str
            The key to save the object under
        """
        path = self.path(key)
        with path.open("wb") as f:
            pickle.dump(picklable, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, key: KT) -> VT:
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
        with path.open("rb") as f:
            return pickle.load(f)

    def __iter__(self) -> Iterator[KT]:
        matches = iter(re.match(self.pattern, file.name) for file in self.dir.iterdir())
        return iter(self.__name_to_key__(match.group(1)) for match in matches if match is not None)
