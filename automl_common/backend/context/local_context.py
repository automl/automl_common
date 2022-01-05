from typing import IO, Iterator, List, Optional

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from automl_common.backend.context import Context, PathLike


class LocalContext(Context):
    """A local context for files on this local machine"""

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
        with open(path, mode=mode) as f:
            yield f

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
    def tmpdir(
        self, prefix: Optional[str] = None, retain: bool = False
    ) -> Iterator[Path]:
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
            The directory path
        """
        path = tempfile.mkdtemp(prefix=prefix)
        yield Path(path)

        if not retain and self.exists(path):
            self.rmdir(path)

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
        return Path(path)
