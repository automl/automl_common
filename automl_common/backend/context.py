"""Defines contexts for file interaction

For now we only support a local context which uses os
"""
from abc import ABC, abstractmethod
from typing import List, IO, Iterator
from contextlib import contextmanager

import os
import shutil
import tempfile

class Context(ABC):
    """A object that lets file operations be performed in some place"""

    @contextmanager
    @abstractmethod
    def open(self, path: str, mode: str) -> Iterator[IO]:
        """A file handle to a given path

        Parameters
        ----------
        path: str
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
    def mkdir(self, path: str) -> None:
        """Make a directory

        Parameters
        ----------
        path: str
            The path to where the directory should be made
        """
        ...

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Recursively make directories, creating those that don't exist one the way

        Parameters
        ----------
        path: str
            The end path to make

        exist_ok: bool = False
            Whether to raise an error if the end directory or any intermediate path
            exists
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Whether a given path exists

        Parameters
        ----------
        path: str
            The path to the file or directory

        Returns
        -------
        bool
            Whether it exists or not
        """
        ...

    @abstractmethod
    def rm(self, path: str) -> None:
        """Delete a file

        Parameters
        ----------
        path: str
            The path to the file to remove
        """
        ...

    @abstractmethod
    def rmdir(self, path: str) -> None:
        """Delete a directory

        Parameters
        ----------
        path: str
            The path to the directory to remove
        """
        ...

    @abstractmethod
    def tmpdir(self, prefix: str) -> str:
        """Return a directory name that can be used as a temporary directory.

        This should not be cleaned up by the Context, only to be used for getting a
        unique name in a valid space.

        Parameters
        ----------
        prefix: str
            A prefix to attach to the directory

        Returns
        -------
        str
            The directory name
        """
        ...

    @abstractmethod
    def join(self, *args: str) -> str:
        """Join parts of path together

        Parameters
        ----------
        *args: str
            Any amount of strings

        Returns
        -------
        str
            The joined path
        """
        ...

    @abstractmethod
    def listdir(self, dir: str) -> List[str]:
        """List the files in a directory

        Parameters
        ----------
        dir: str
            The directory to list

        Returns
        -------
        List[str]
            The folders and files in a directory
        """

class LocalContext(Context):
    """A local context for files, using `os` and `shutil`
    """

    def open(self, path: str, mode: str) -> Iterator[IO]:
        """A file handle to a given path

        Parameters
        ----------
        path: str
            A path to the file

        mode: str = 'r'
            One of mode flags used for python's `open`. See
            `https://docs.python.org/3/library/functions.html#open`_

        Returns
        -------
        IO
            Returns a file object that is opened in the associated mode
        """
        yield open(path=path, mode=mode)

    def mkdir(self, path: str) -> None:
        """Make a directory

        Parameters
        ----------
        path: str
            The path to where the directory should be made
        """
        os.mkdir(path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Recursively make directories, creating those that don't exist one the way

        Parameters
        ----------
        path: str
            The end path to make

        exist_ok: bool = False
            Whether to raise an error if the end directory or any intermediate path
            exists
        """
        os.makedirs(path=path, exist_ok=exist_ok)

    def exists(self, path: str) -> bool:
        """Whether a given path exists

        Parameters
        ----------
        path: str
            The path to the file or directory

        Returns
        -------
        bool
            Whether it exists or not
        """
        return os.path.exists(path)

    def rm(self, path: str) -> None:
        """Delete a file

        Parameters
        ----------
        path: str
            The path to the file to remove
        """
        os.remove(path)

    def rmdir(self, path: str) -> None:
        """Delete a directory

        Parameters
        ----------
        path: str
            The path to the directory to remove
        """
        shutil.rmtree(path)

    def tmpdir(self, prefix: str) -> str:
        """Return a directory name that can be used as a temporary directory.

        This should not be cleaned up by the Context, only to be used for getting a
        unique name in a valid space.

        Parameters
        ----------
        prefix: str
            A prefix to attach to the directory

        Returns
        -------
        str
            The directory name
        """
        return tempfile.mkdtemp(prefix=prefix)

    def join(self, *args: str) -> str:
        """Join parts of path together

        Parameters
        ----------
        *args: str
            Any amount of strings

        Returns
        -------
        str
            The joined path
        """
        return os.path.join(*args)

class AWSContext(Context):
    """A Context for AWS ... just as example of what other contexts could exist """

    def __init__(self, some_key: str):
        raise NotImplementedError()
