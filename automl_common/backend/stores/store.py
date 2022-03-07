from abc import ABC, abstractmethod
from typing import Iterator, Mapping, MutableMapping, TypeVar

from pathlib import Path

from automl_common.backend.util.path import rmtree
from automl_common.util.types import EqualityMixin

KT = TypeVar("KT")
VT = TypeVar("VT")


class StoreView(ABC, EqualityMixin, Mapping[KT, VT]):
    """An immutable view into state preserved on the filesystem

    Stores items by key onto the filesystem but can not write to
    the filesystem itself

    Implementers must satisfy:
    * `load`        - Load an object VT given a key
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

    def __getitem__(self, key: KT) -> VT:
        if key not in self:
            raise KeyError(f"No item with {key} found in {self.dir}")

        return self.load(key)

    def __contains__(self, key: object) -> bool:
        # Can't do an isinstance check as KT could be a complicated type
        try:
            path = self.path(key)  # type: ignore
            return path.exists()
        except Exception:
            return False

    def __len__(self) -> int:
        return len(list(self.__iter__()))

    def path(self, key: KT) -> Path:
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
        return self.dir / self.__key_to_str__(key)

    def __iter__(self) -> Iterator[KT]:
        return map(self.__name_to_key__, iter(path.name for path in self.dir.iterdir()))

    def iterpaths(self) -> Iterator[Path]:
        """Iterator over the paths in the store

        Returns
        -------
        Iterator[Path]
        """
        return iter(self.path(key) for key in self)

    @abstractmethod
    def load(self, key: KT) -> VT:
        """Loads an object from the path

        Parameters
        ----------
        key: str
            The keyed object to load

        Returns
        -------
        VT
            The loaded object
        """
        ...

    def __repr__(self) -> str:
        return f"Store: {self.dir}"

    def __key_to_str__(self, key: KT) -> str:
        return str(key)

    def __name_to_key__(self, path_name: str) -> KT:
        """Convert the name at the end of a path to a key

        For example, given the path /x/y/zebra_dog.npy, __name_to_key__ will be
        passed "zebra_dog.npy" and need to convert appropriatly

        Note
        ----
        Default behaviour assumes KT is a string but typing isn't correct if it's not.
        Implementers would implement this themselves for KT, we can't provide a default
        while keeping it type safe.

        Parameters
        ----------
        path_name : str
            The path name to convert

        Returns
        -------
        KT
            The converted key
        """
        # We just assume string, implementers get nice default behaviour but need
        # to implement it themselves for anything more complicated.
        return str(path_name)  # type: ignore


class Store(StoreView[KT, VT], MutableMapping[KT, VT]):
    """An mutable view into state preserved on the filesystem

    Stores items by key onto the filesystem and can write items
    to the filesystem

    An extending class must implement the
    * `save`        - Save an object to the store
    * `load`        - Load an object to the store
    """

    def __setitem__(self, key: KT, obj: VT) -> None:
        self.save(obj, key)

    def __delitem__(self, key: KT) -> None:
        if key not in self:
            raise KeyError(f"No item with {key} found at {self.path(key)}")

        path = self.path(key)
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    @abstractmethod
    def save(self, obj: VT, key: KT) -> None:
        """Saves the object as key

        Parameters
        ----------
        obj: VT
            The object to sae

        key: KT
            The key to save it under
        """
        ...
