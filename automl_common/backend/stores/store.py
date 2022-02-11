from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping, MutableMapping, TypeVar

from pathlib import Path

from automl_common.backend.util.path import rmtree
from automl_common.util.types import EqualityMixin

T = TypeVar("T")


class StoreView(ABC, EqualityMixin, Mapping[str, T]):
    """An immutable view into state preserved on the filesystem

    Stores items by key onto the filesystem but can not write to
    the filesystem itself

    Implementers must satisfy:
    * `load`        - Load an object T given a key
    """

    def __init__(self, dir: Path):
        """
        Parameters
        ----------
        dir: Path
            The directory to load and store from
        """
        self.dir = dir
        if not dir.exists():
            dir.mkdir()

    def __getitem__(self, key: str) -> T:
        if key not in self:
            raise KeyError(f"No item with {key} found at {self.path(key)}")

        return self.load(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.path(key).exists()

    def __len__(self) -> int:
        return len(list(self.__iter__()))

    def path(self, key: str) -> Path:
        """Get the fullpath to an object stored with key

        Parameters
        ----------
        key: str
            The key under which it was stored

        Returns
        -------
        Path
            The path to an object with a given key
        """
        return self.dir / key

    def __iter__(self) -> Iterator[str]:
        return iter(path.name for path in self.dir.iterdir())

    @abstractmethod
    def load(self, key: str) -> T:
        """Loads an object from the path

        Parameters
        ----------
        key: str
            The keyed object to load

        Returns
        -------
        T
            The loaded object
        """
        ...

    def __repr__(self) -> str:
        return f"Store: {self.dir}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, StoreView) and self.dir == other.dir


class Store(StoreView[T], MutableMapping[str, T]):
    """An mutable view into state preserved on the filesystem

    Stores items by key onto the filesystem and can write items
    to the filesystem

    An extending class must implement the
    * `save`        - Save an object to the store
    * `load`        - Load an object to the store
    """

    def __setitem__(self, key: str, obj: T) -> None:
        self.save(obj, key)

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(f"No item with {key} found at {self.path(key)}")

        path = self.path(key)
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    @abstractmethod
    def save(self, obj: T, key: str) -> None:
        """Saves the object as key

        Parameters
        ----------
        obj: T
            The object to sae

        key: str
            The key to save it under
        """
        ...
