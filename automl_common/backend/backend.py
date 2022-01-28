from typing import Optional, Union

import tempfile
from pathlib import Path


class Backend:
    """Manages the a directory and ensures clean up after deletion"""

    def __init__(
        self,
        name: str,
        path: Optional[Union[str, Path]] = None,
        retain: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        name: str
            The name of to give this backend

        path: Optional[Union[str, Path]] = None
            A path where the backend will be rooted. If None is provided, we assume
            a local context and create a tmp path

        retain: Optional[bool] = None
            Whether to retain the folder once the backend has been garbage collected.
            If left as None, this will delete any tmpdir allocated with `path` left as
            None, otherwise, if a path is specified, it will retain it.
        """
        if path is None:
            self.path = Path(tempfile.mkdtemp(prefix=name))
        elif isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path

        # We default to retaining if a path was specified
        if retain is None:
            self.retain = False if path is None else True
        else:
            self.retain = retain

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

    def __del__(self) -> None:
        """Delete the folders if we do not retain them."""
        if not self.retain and self.path.exists():
            self.path.rmdir()
