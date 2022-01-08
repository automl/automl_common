"""
Defines contexts for file interaction
For now we only support a local context which uses os
"""
from abc import ABC, abstractmethod
from typing import IO, Iterator, List, Optional, Union

from contextlib import contextmanager
from pathlib import Path

PathLike = Union[str, Path]


class Context(ABC):
    """An `os` like class to interact with a filesystem. It additionally manages
    cleanup and temporary direcory cleanup if required.
    """

    @classmethod
    @contextmanager
    def tmpdir(
        cls,
        prefix: Optional[str] = None,
        retain: bool = False,
    ) -> Iterator[Path]:
        """Return a directory path as a context manager

        Parameters
        ----------
        prefix: Optional[str] = None
            A prefix to attach to the directory

        retain: bool = False
            Whether to keep the directory after the context ends

        Returns
        -------
        Iterator[Path]
            The directory path
        """
        path = cls.mkdtemp(prefix=prefix)
        yield Path(path)

        if not retain and cls.exists(path):
            cls.rmdir(path)

    @classmethod
    @contextmanager
    @abstractmethod
    def open(cls, path: PathLike, mode: str = "r") -> Iterator[IO]:
        """A file handle to a given path

        Parameters
        ----------
        path: PathLike
            A path to the file

        mode: str = 'r'
            One of mode flags used for python's `open`. See
            `https://docs.python.org/3/library/functions.html#open`_

        Returns
        -------
        IO
            Returns a file object that is opened in the associated mode
        """
        ...

    @classmethod
    @abstractmethod
    def mkdir(cls, path: PathLike) -> None:
        """Make a directory

        Parameters
        ----------
        path: PathLike
            The path to where the directory should be made
        """
        ...

    @classmethod
    @abstractmethod
    def makedirs(cls, path: PathLike, exist_ok: bool = False) -> None:
        """Recursively make directories, creating those that don't exist one the way

        Parameters
        ----------
        path: PathLike
            The end path to make

        exist_ok: bool = False
            Whether to raise an error if the end directory or any intermediate path
            exists
        """
        ...

    @classmethod
    @abstractmethod
    def exists(cls, path: PathLike) -> bool:
        """Whether a given path exists

        Parameters
        ----------
        path: PathLike
            The path to the file or directory

        Returns
        -------
        bool
            Whether it exists or not
        """
        ...

    @classmethod
    @abstractmethod
    def rm(cls, path: PathLike) -> None:
        """Delete a file

        Parameters
        ----------
        path: PathLike
            The path to the file to remove
        """
        ...

    @classmethod
    @abstractmethod
    def rmdir(cls, path: PathLike) -> None:
        """Delete a directory

        Parameters
        ----------
        path: PathLike
            The path to the directory to remove
        """
        ...

    @classmethod
    @abstractmethod
    def mkdtemp(cls, prefix: Optional[str] = None) -> Path:
        """Create a temporary folder the user is responsible for deleting

        Parameters
        ----------
        prefix: Optional[str]
            A prefix to add to the tmp folder name that is created

        Returns
        -------
        Path
            The path to the tmp folder
        """
        ...

    @classmethod
    @abstractmethod
    def listdir(cls, dir: PathLike) -> List[str]:
        """List the files in a directory

        Parameters
        ----------
        dir: PathLike
            The directory to list

        Returns
        -------
        List[str]
            The folders and files in a directory
        """
        ...

    @classmethod
    @abstractmethod
    def as_path(cls, path: str) -> Path:
        """Convert a str path to a Path object used for this context

        Parameters
        ----------
        path: str
            The path as a raw str

        Returns
        -------
        Path
            An object following the Path interface
        """
        ...
