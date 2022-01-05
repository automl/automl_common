"""Defines contexts for file interaction

For now we only support a local context which uses os
"""
from abc import ABC, abstractmethod
from typing import IO, Iterator, List, Optional, Union

import os
from contextlib import contextmanager
from pathlib import Path

PathLike = Union[str, os.PathLike]


class Context(ABC):
    """A object that lets file operations be performed in some place"""

    @contextmanager
    @abstractmethod
    def open(self, path: PathLike, mode: str) -> Iterator[IO]:
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

    @abstractmethod
    def mkdir(self, path: PathLike) -> None:
        """Make a directory

        Parameters
        ----------
        path: PathLike
            The path to where the directory should be made
        """
        ...

    @abstractmethod
    def makedirs(self, path: PathLike, exist_ok: bool = False) -> None:
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

    @abstractmethod
    def exists(self, path: PathLike) -> bool:
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

    @abstractmethod
    def rm(self, path: PathLike) -> None:
        """Delete a file

        Parameters
        ----------
        path: PathLike
            The path to the file to remove
        """
        ...

    @abstractmethod
    def rmdir(self, path: PathLike) -> None:
        """Delete a directory

        Parameters
        ----------
        path: PathLike
            The path to the directory to remove
        """
        ...

    @contextmanager
    @abstractmethod
    def tmpdir(
        self, prefix: Optional[str] = None, retain: bool = False
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
        ...

    @abstractmethod
    def listdir(self, dir: PathLike) -> List[str]:
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

    @abstractmethod
    def as_path(self, path: str) -> Path:
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
