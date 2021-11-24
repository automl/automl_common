from typing import Dict, List, Optional, Tuple, TypeVar, Generic, Union, cast, IO
from contextlib import contextmanager

import glob
import os
import pickle
import shutil
import tempfile
import time
import warnings

import numpy as np

from sklearn.pipeline import Pipeline

from automl_common.logging_ import PicklableClientLogger, get_named_client_logger
from automl_common.ensemble_building.abstract_ensemble import AbstractEnsemble
from automl_common.backend.context import Context, LocalContext
from automl_common.backend.optimizer import Optimizer
from automl_common.backend.model import Model
from automl_common.backend.datamanager import DataManager
from automl_common.backend.ensemble import Ensemble, Ensembles
from automl_common.backend.run import Run


Model = TypeVar("Model")  # The Type of Model loaded
RunID = TypeVar("RunID")  # The Type of identifier for a unique run
EnsembleID = TypeVar("EnsembleID")  # The Type of identifier for a unique ensemble
DM = TypeVar("DM")  # The Type of the datamanager

class Backend(Generic[Model], Generic[EnsembleID], Generic[RunID], Generic[DM]):
    """Utility class to load and save objects to be persisted

    A backend is parameterized by 4 Types
    * Model - The Type of Model loaded
    * RunID - The Type of identifier for a unique run
    * EnsembleID -  The Type of identifier for a unique ensemble
    * DM -  The Type of the datamanager


    # An example of a backend where it returns models of `MyModelType`, where each
    # run is identified by an `int`, and each ensemble created is identified by
    # an `int`. Finally, the kind of datamanger used is `MyDataManager`

    backend: [MyModelType, int, int, MyDataManager] = Backend(...)
    """

    def __init__(
        self,
        framework: str,
        root: str,
        context: Optional[Context] = None,
        retain: bool = False
    ):
        """
        Parameters
        ----------
        framework: str
            The name of the framework

        root: str
            The root directory of this backend

        context: Optional[Context] = None
            The context to operate in. For now only a local context operating on the
            local filesystem is supported.

        retain: bool = False
            Whether to keep the backend and it's content after the object has unloaded.
        """
        if context is None:
            context = LocalContext()

        if context.exists(root):
            raise ValueError(f"Path {root} already exists, reusing paths is not supported yet")

        self.root = root
        self.framework = framework
        self.context = context
        self.retain = retain

        self._optimizer = Optimizer(dir=self.optimizer_dir, context=context)
        self._ensembles = Ensembles(dir=self.ensembles_dir, context=context)
        self._datamanager: DataManager[DM] = DataManager(self.data_dir, context=context)

        self._logger: Optional[PicklableClientLogger] = None

        # Create the folders we can control, users may decide to create their own
        # extra folders. We have flexible way to manage these other than they are under
        # root
        self.context.mkdir(self.root)
        folders = [
            self.framework_dir,
            self.optimizer_dir,
            self.ensembles_dir,
            self.runs_dir,
            self.data_dir
        ]
        for folder in folders:
            path = self.context.join(self.root, folder)
            self.context.mkdir(path)

    def __del__(self):
        """Delete the folders if we do not retain them."""
        if not self.retain:
            self.context.rmdir(self.root)

    @property
    def logger(self) -> PicklableClientLogger:
        """Logger for this backend, creates one if hasn't been created yet"""
        if self._logger is None:
            self._logger = get_named_client_logger(name=__name__)

        return self._logger

    def setup_logger(self, port: int) -> None:
        """Set up the logger with the given port once we are aware of it.

        Parameters
        ----------
        port: int
            Which port the logger should be hosted on
        """
        self.logger = get_named_client_logger(name=__name__, port=port)

    @property
    def framework_dir(self) -> str:
        """Directory for framework specific files."""
        return self.context.join(self.root, self.framework)

    @property
    def optimizer_dir(self) -> str:
        """Directory for optimizer specific files."""
        return self.context.join(self.root, "optimizer")

    @property
    def ensembles_dir(self) -> str:
        """Directory for ensemble related files"""
        return self.context.join(self.root, "ensembles")

    @property
    def runs_dir(self) -> str:
        """Directory for all the runs"""
        return self.context.join(self.root, "runs")

    @property
    def data_dir(self) -> str:
        """Directory for data specific files"""
        return self.context.join(self.root, "data")

    @property
    def start_time_path(self) -> str:
        """Path to where the start time is written"""
        return self.context.join(self.root, "start_time.marker")

    @property
    def end_time_path(self) -> str:
        """Path to where the end time is written"""
        return self.context.join(self.root, "end_time.marker")

    @property
    def ensembles(self) -> Ensembles:
        """Object to access to the ensembles part of the backend"""
        return self._ensembles

    @property
    def optimizer(self) -> Optimizer:
        """Object to access to the optimizer part of the backend"""
        return self._optimizer

    @property
    def data(self) -> DataManager[DM]:
        """Object to access to the data part of the backend"""
        return self._datamanager

    def run(self, id: RunID) -> Run[RunID, Model]:
        """Object to access files and directories for a specific run

        Parameters
        ----------
        id: RunID
            The identifier for the run

        Returns
        -------
        Run[RunID, Model]
            A run object to access files and directories for a specific run
        """
        return Run(id=id, root=self.runs_dir, context=self.context)

    def ensemble(self, id: EnsembleID) -> Ensemble[EnsembleID]:
        """Object to access files and directories for a specific ensemble

        Parameters
        ----------
        id: EnsembleID
            The identifier for the ensemble

        Returns
        -------
        Ensemble[EnsembleID]
            A run object to access files and directories for a specific run
        """
        return Ensemble(id=id, root=self.ensembles_dir)

    def mark_start(self) -> None:
        """Write the start time to file"""
        start_time = str(time.time())
        with self.context.open(self.start_time_path, "w") as f:
            f.write(start_time)

    def mark_end(self) -> None:
        """Write the end time to file"""
        end_time = str(time.time())
        with self.context.open(self.end_time_path, "w") as f:
            f.write(end_time)

    def runs(self) -> List[str]:
        """A list of run names in the runs folder"""
        return os.listdir(self.runs_dir)

    def models(self, ids: List[RunID]) -> List[Model]:
        """A list of Models gotten by their RunID

        Parameters
        ----------
        ids: List[RunID]
            The list of models to get

        Returns
        -------
        List[Model]
            A list of models retrieved
        """
        return [self.run(id).model() for id in ids]

    @contextmanager
    def open(self, path: str, mode: str) -> Iterator[IO]:
        """Open a file

        # NOTE:
        #   User would have no way to join files. Not an issue in local setting, might
        #   be an issue if using something like AWS filesystem where os.join does not
        #   match

        Parameters
        ----------
        path: str
            The path to the file

        mode: str
            One of the modes that can be passed to file opening

        Returns
        -------
        Iterator[IO]
            A context manager that can be used in the normal way
            ```
            with backend.open(...) as f:
                ...
            ```
        """
        yield self.context.open(path, mode)
