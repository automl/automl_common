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
from automl_common.backend.run import Run, Runs


Model = TypeVar("Model")  # The Type of Model loaded
DM = TypeVar("DM")  # The Type of the datamanager

class Backend(Generic[Model], Generic[DM]):
    """Utility class to load and save objects to be persisted

    A backend is parameterized by 2 Types
    * Model - The Type of Model loaded
    * DM -  The Type of the datamanager

    backend: [MyModelType, MyDataManager] = Backend(...)

    Id's used for ensembles are str. The reason for this is that we do not know what
    a framework would like to use as a key and these keys must be written as a directory
    name. Any id may be used as long as it can be converted to a string. As we load
    identifiers from directory names, they are internally stored as str.

    tldr;
    ```
    backend.runs[1337].exists() # False

    id = (3,4,5)
    type(id) # tuple                    - (3,4,5)
    type(backend.runs[id].id) # str     - "(3,4,5)"

    # Existence and creation
    backend.runs[id].exists() # False

    backend.runs[id].save_model(my_model) # Save something
    backend.runs[id].exists() # True

    # Save predictions
    backend.runs[id].save_predictions(preds, prefix="train")

    # Accessing things
    model = backend.runs[id].model()
    train_predictions = backend.runs[id].predictions("train")

    # Works as a map
    for id, run in backend.runs.items():
        predictions = run.predictions()

    # Some usages
    # Supposing we have a directory of 10 runs id's from 1, 10

    # Getting specific models
    ids = set(1, 2, 3, 4)
    models = [run.model() for id, run in backend.runs.items() if id in ids]
    predictions = [run.predictions("train") for id, run in backend.runs.items() if id in ids]

    # Some checking information
    len(backend.runs) # 10
    list(backend.runs) # [1, ..., 10]
    6 in backend.runs # True

    # Generate predictions for all models in the backend
    model_predictions = {
        id: run.model().predict(X_test)
        for id, run in backend.runs.items
    }
    ```

    Backend has a similar object `ensembles` which works in much the same way

    i.e. `backend.ensembles`

    The framework dir and optimizer dir are setup to be used as you would like.

    The runs dir holds all the runs, labeled by an id and containing a run and it's
    predictions.

    /<root>
        /<framework>
            - ...
        /optimizer
        /data
            - datamanger.npy
        /ensembles
            - targets.npy
            /<ensembleid>
                - ensemble
                - ...
            /<ensembleid>
                - ensemble
                - ...
        /runs
            /<runid>
                - model
                - <prefix>_predictions
                - ...
            /<runid>
                - model
                - <prefix>_predictions
                - ...
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
        self._runs = Runs(dir=self.runs_dir, context=context)
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
    def runs(self) -> Runs:
        """Object to access the runs part of the backend"""
        return self._runs

    @property
    def optimizer(self) -> Optimizer:
        """Object to access to the optimizer part of the backend"""
        return self._optimizer

    @property
    def data(self) -> DataManager[DM]:
        """Object to access to the data part of the backend"""
        return self._datamanager

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

    def models(self, ids: List[str]) -> List[Model]:
        """A list of Models gotten by their id

        Parameters
        ----------
        ids: List[str]
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
