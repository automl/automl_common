from abc import ABC, abstractmethod
from typing import Iterator, Mapping, MutableMapping, TypeVar

from pathlib import Path

from automl_common.backend.contexts.context import Context, PathLike

T = TypeVar("T")


class StoreView(ABC, Mapping[str, T]):
    """An immutable view into state preserved on the filesystem

    Stores items by key onto the filesystem but can not write to
    the filesystem itself

    Implementers must satisfy:
    * `load`        - Load an object T given a key
    * `__iter__`    - Iterate over keys in the store
    """

    def __init__(self, dir: PathLike, context: Context):
        """
        Parameters
        ----------
        dir: PathLike
            The directory to load and store from

        context: Context
            A context object to iteract with a filesystem
        """
        self.dir
        self.context = context
        self.dir: Path

        if isinstance(dir, Path):
            self.dir = dir
        else:
            self.dir = self.context.as_path(dir)

        if not self.context.exists(dir):
            self.context.mkdir(dir)

    def __getitem__(self, key: str) -> T:
        return self.load(key)

    def __contains__(self, key: object) -> bool:
        # Note why is it typed as object
        # https://github.com/python/mypy/issues/5633#issuecomment-422434923
        if isinstance(key, str):
            path = self.path(key)
            return self.context.exists(path)
        else:
            return False

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

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...

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


class Store(StoreView[T], MutableMapping[str, T]):
    """An mutable view into state preserved on the filesystem

    Stores items by key onto the filesystem and can write items
    to the filesystem

    An extending class must implement the
    * `save`        - Save an object to the store
    * `load`        - Load an object to the store
    * `__iter__`    - Iterate over keys in the store
    """

    def __setitem__(self, key: str, obj: T) -> None:
        self.save(obj, key)

    def __delitem__(self, key: str) -> None:
        path = self.path(key)
        self.context.rm(path)

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
