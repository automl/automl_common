from typing import IO, Generic, Iterator, List, Optional, TypeVar

import time
from contextlib import contextmanager
from pathlib import Path

from ..utils.logging_ import PicklableClientLogger, get_named_client_logger
from .context import Context, LocalContext, PathLike
from .datamanager import DataManager
from .ensembles import Ensembles
from .optimizer import Optimizer
from .runs import Runs

Model = TypeVar("Model")  # The Type of Model loaded
DM = TypeVar("DM")  # The Type of the datamanager


class Backend(Generic[Model, DM], Context):
    """Utility class to load and save objects to be persisted

    Note
    ----
        Inheriting from Context as this provides all the functionality of one,
        all be it by wrapping another context and just forwarding all its
        methods.

    A backend is parameterized by 2 Types
    * Model - The Type of Model loaded
    * DM -  The Type of the datamanager

    backend: Backend[MyModelType, MyDataManager] = Backend(...)

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
    predictions = [
        run.predictions("train") for id, run in backend.runs.items() if id in ids
    ]

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
        framework: str = "framework",
        root: Optional[str] = None,
        context: Optional[Context] = None,
        retain: bool = False,
    ):
        """
        Parameters
        ----------
        framework: str = "framework"
            The name of the framework. Defaults to "framework"

        root: Optional[str] = None
            The root directory of this backend. Defaults to the contexts tmpdir.

        context: Optional[Context] = None
            The context to operate in. For now only a local context operating on the
            local filesystem is supported. Defaults to a `LocalContext()`

        retain: bool = False
            Whether to keep the backend and it's content after the object has unloaded.
        """
        if context is None:
            context = LocalContext()

        # Make sure the root directory doesn't exist
        if root is not None:
            if context.exists(root):
                raise RuntimeError(
                    f"{root} already exists, does not support reusing directories yet"
                )
            context.mkdir(root)
        else:
            # Otherwise we create a tmpdir as our root
            # We set `retain = True` as we make sure to delete it when this
            # object is unloaded through __del__
            with context.tmpdir(prefix=framework, retain=True) as tmpdir:
                root = tmpdir

        self._root = root
        self._framework = framework
        self._context = context
        self._retain = retain

        self._logger: Optional[PicklableClientLogger] = None

        # Backend objects
        self.optimizer = Optimizer(dir=self.optimizer_dir, context=context)
        self.ensembles = Ensembles(dir=self.ensembles_dir, context=context)
        self.runs = Runs[Model](dir=self.runs_dir, context=context)
        self.datamanager = DataManager[DM](self.data_dir, context=context)

        # Create the folders we can control, users may decide to create their own
        # extra folders. We have flexible way to manage these other than they are under
        folders = [
            self.framework_dir,
            self.optimizer_dir,
            self.ensembles_dir,
            self.runs_dir,
            self.data_dir,
        ]
        for folder in folders:
            path = self.join(self._root, folder)
            self.mkdir(path)

    def __del__(self):
        """Delete the folders if we do not retain them."""
        if not self._retain and self.exists(self._root):
            self.rmdir(self._root)

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
        self._logger = get_named_client_logger(name=__name__, port=port)

    @property
    def framework_dir(self) -> str:
        """Directory for framework specific files."""
        return self.join(self._root, self._framework)

    @property
    def optimizer_dir(self) -> str:
        """Directory for optimizer specific files."""
        return self.join(self._root, "optimizer")

    @property
    def ensembles_dir(self) -> str:
        """Directory for ensemble related files"""
        return self.join(self._root, "ensembles")

    @property
    def runs_dir(self) -> str:
        """Directory for all the runs"""
        return self.join(self._root, "runs")

    @property
    def data_dir(self) -> str:
        """Directory for data specific files"""
        return self.join(self._root, "data")

    @property
    def start_time_path(self) -> str:
        """Path to where the start time is written"""
        return self.join(self._root, "start_time.marker")

    @property
    def end_time_path(self) -> str:
        """Path to where the end time is written"""
        return self.join(self._root, "end_time.marker")

    def mark_start(self) -> None:
        """Write the start time to file"""
        if self.exists(self.start_time_path):
            raise RuntimeError(f"Already started, written in {self.start_time_path}")

        start_time = str(time.time())
        with self.open(self.start_time_path, "w") as f:
            f.write(start_time)

    def mark_end(self) -> None:
        """Write the end time to file"""
        if self.exists(self.end_time_path):
            raise RuntimeError(f"Already started, written in {self.end_time_path}")
        end_time = str(time.time())
        with self.open(self.end_time_path, "w") as f:
            f.write(end_time)

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
        with self._context.open(path, mode) as f:
            yield f

    def mkdir(self, path: PathLike) -> None:
        """Make a directory

        Parameters
        ----------
        path: PathLike
            The path to where the directory should be made
        """
        self._context.mkdir(path)

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
        self._context.makedirs(path, exist_ok=exist_ok)

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
        return self._context.exists(path)

    def rm(self, path: PathLike) -> None:
        """Delete a file

        Parameters
        ----------
        path: PathLike
            The path to the file to remove
        """
        self._context.rm(path)

    def rmdir(self, path: PathLike) -> None:
        """Delete a directory

        Parameters
        ----------
        path: PathLike
            The path to the directory to remove
        """
        self._context.rmdir(path)

    @contextmanager
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
        with self._context.tmpdir(prefix=prefix, retain=retain) as tmpdir:
            yield tmpdir

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
        return self._context.join(*args)

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
        return self._context.listdir(dir)
