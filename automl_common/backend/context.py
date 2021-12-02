"""Defines contexts for file interaction

For now we only support a local context which uses os
"""
from abc import ABC, abstractmethod
from typing import List, IO, Iterator, Union, Optional
from contextlib import contextmanager

import os
import shutil
import tempfile
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
    def tmpdir(self, prefix: Optional[str] = None, retain: bool = False) -> Iterator[Path]:
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
    def join(self, *args: PathLike) -> str:
        """Join parts of path together

        Parameters
        ----------
        *args: PathLike
            Any amount of PathLike

        Returns
        -------
        str
            The joined path
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


class LocalContext(Context):
    """A local context for files, using `os` and `shutil`
    """

    @contextmanager
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
        yield open(path, mode=mode)

    def mkdir(self, path: PathLike) -> None:
        """Make a directory

        Parameters
        ----------
        path: PathLike
            The path to where the directory should be made
        """
        os.mkdir(path)

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
        os.makedirs(path, exist_ok=exist_ok)

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
        return os.path.exists(path)

    def rm(self, path: PathLike) -> None:
        """Delete a file

        Parameters
        ----------
        path: PathLike
            The path to the file to remove
        """
        os.remove(path)

    def rmdir(self, path: PathLike) -> None:
        """Delete a directory

        Parameters
        ----------
        path: PathLike
            The path to the directory to remove
        """
        shutil.rmtree(path)

    @contextmanager
    def tmpdir(self, prefix: Optional[str] = None, retain: bool = False) -> Iterator[Path]:
        """Return a temporary directory

        `with context.tmpdir() as tmpdir: ...`

        Parameters
        ----------
        prefix: Optional[str] = None
            A prefix to attach to the directory

        retain: bool = False
            Whether to retain the directory after the context finishes

        Returns
        -------
        Iterator[Path]
            The directory name
        """
        path = Path(tempfile.mkdtemp(prefix=prefix))
        yield path

        if not retain and self.exists(path):
            self.rmdir(path)

    def join(self, *args: PathLike) -> str:
        """Join parts of path together

        Parameters
        ----------
        *args: PathLike
            Any amount of path-likes

        Returns
        -------
        str
            The joined path. The type is in line with `os.path.join`
        """
        return os.path.join(*args)

    def listdir(self, path: PathLike) -> List[str]:
        """Return a list of the contents of a directory

        Parameters
        ----------
        path: PathLike
            The path to the directory to list

        Returns
        -------
        List[str]
            A list of the contents of the directory
        """
        return os.listdir(path)


class AWSContext(Context):
    """A Context for AWS ... just as example of what other contexts could exist """

    def __init__(self, some_key: str):
        raise NotImplementedError()
