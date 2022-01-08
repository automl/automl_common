from typing import Optional

from automl_common.backend.backend import Backend, LocalBackend, PathLike


class AutoMLBackend:
    def __init__(
        self, path: Optional[PathLike] = None, backend: Optional[Backend] = None
    ):
        """
        Parameters
        ----------
        path: Optional[PathLike] = None
            The path to store files to

        backend: Optional[Backend] = None
            The backend to use, for now only LocalBackend is supported
        """
        if not isinstance(backend, LocalBackend):
            raise NotImplementedError()

        self.backend = LocalBackend(framework="autosklearn", root=path)
