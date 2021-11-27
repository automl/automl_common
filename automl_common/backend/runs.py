from collections.abc import Mapping
from typing import Any, Iterable

from automl_common.backend.context import Context
from automl_common.backend.run import Run


class Runs(Mapping):
    """Interaface to the runs directory in the backend

    /<dir>
        /<id>
            - model
            - {prefix}_predictions
        /<id>
            - model
            - {prefix}_predictions
        /...
    """

    def __init__(self, dir: str, context: Context):
        """
        Parameters
        ----------
        dir: str
            The directory of the runs

        context: Context
            The context to access the filesystem through
        """
        self.dir = dir
        self.context = context

    def __getitem__(self, id: Any) -> Run:
        """Get a run

        Parameters
        ----------
        id: Any
            The id of the run
        """
        run_dir = self.context.join(self.dir, str(id))
        return Run(id=str(id), dir=run_dir, context=self.context)

    def __iter__(self) -> Iterable[str]:
        """Iterate over runs

        Returns
        -------
        Iterable[str]
            Key, value pairs of identifiers to Run objects
        """
        return (dir for dir in self.context.listdir(self.dir))

    def __contains__(self, id: Any) -> bool:
        """Whether a given run is contained in the backend

        Parameters
        ----------
        id: Any
            The id of the run to get

        Returns
        -------
        bool
            Whether this run is contained in the backend
        """
        path = self.context.join(self.dir, str(id))
        return self.context.exists(path)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The amount of runs in the backend
        """
        return len(self.context.listdir(self.dir))
