from typing import IO, Iterator, List, Optional

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from automl_common.backend.contexts import Context, PathLike


class OSContext(Context):
    """A local context for files on this local machine"""

    @classmethod
    @contextmanager
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
        with open(path, mode=mode) as f:
            yield f

    @classmethod
    def mkdir(cls, path: PathLike) -> None:
        """Make a directory

        Parameters
        ----------
        path: PathLike
            The path to where the directory should be made
        """
        os.mkdir(path)

    @classmethod
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
        os.makedirs(path, exist_ok=exist_ok)

    @classmethod
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
        return os.path.exists(path)

    @classmethod
    def rm(cls, path: PathLike) -> None:
        """Delete a file

        Parameters
        ----------
        path: PathLike
            The path to the file to remove
        """
        os.remove(path)

    @classmethod
    def rmdir(cls, path: PathLike) -> None:
        """Delete a directory

        Parameters
        ----------
        path: PathLike
            The path to the directory to remove
        """
        shutil.rmtree(path)

    @classmethod
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
        path = tempfile.mkdtemp(prefix=prefix)
        return Path(path)

    @classmethod
    def listdir(cls, path: PathLike) -> List[str]:
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

    @classmethod
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
        return Path(path)
