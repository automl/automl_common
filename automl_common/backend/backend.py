from typing import Generic, Optional, TypeVar

from pathlib import Path

from automl_common.backend.contexts import Context, OSContext, PathLike
from automl_common.backend.stores.ensemble_store import EnsembleStore
from automl_common.backend.stores.model_store import ModelStore
from automl_common.model import Model

ModelT = TypeVar("ModelT", bound=Model)


class Backend(Generic[ModelT]):
    """Manages general model access along with whatever else automl_common can provide

    Optimizers can also use the backend as desired

    Manages:
    /<path>
        /models
            / <key>
                / predictions_{}.npy
                / model
                / ...
            / <key>
                / predictions_{}.npy
                / model
                / ...
        /ensembles
            / <key>
                / predictions_{}.npy
                / ensemble
                / ...
            / <key>
                / predictions_{}.npy
                / ensemble
                / ...
        /...
    """

    def __init__(
        self,
        name: str,
        path: Optional[PathLike] = None,
        context: Optional[Context] = None,
        retain: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        name: str
            The name of to give this backend

        path: Optional[PathLike] = None
            A path where the backend will be rooted. If None is provided, one will
            be allocated in the contexts tmp directory.

        context: Optional[Context] = None
            A context to perform operations to a filesystem similar to the `os` module.
            If None is provided, it will default to the local system and use a wrapped
            version of Python's `os`.

        retain: Optional[bool] = None
            Whether to retain the folder once the backend has been garbage collected.
            If left as None, this will delete any tmpdir allocated with `path` left as
            None, otherwise, if a path is specified, it will retain it.
        """
        if context is not None and not isinstance(context, OSContext):
            raise NotImplementedError()

        if path is not None:
            self.context = OSContext()
            self.retain = retain if retain is not None else True
            self.path = path if isinstance(path, Path) else self.context.as_path(path)
        else:
            self.context = OSContext()
            self.path = self.context.mkdtemp(prefix=name)
            self.retain = retain if retain is not None else False

        self.name = name

        self._model_store = ModelStore[ModelT](dir=self.model_dir, backend=self)
        self._ensembles_store = EnsembleStore(dir=self.ensemble_dir, backend=self)

    @property
    def model_dir(self) -> Path:
        """Path to the models dir"""
        return self.path / "models"

    @property
    def ensemble_dir(self) -> Path:
        """Path to the ensembles dir"""
        return self.path / "ensembles"

    @property
    def models(self) -> ModelStore[ModelT]:
        """Get dictionary like access to models stored

        Returns
        -------
        ModelStore[Model]
            A store of models that can be used in similar fashion to a
            dict where keys are model names and values are ModelAccessor
            objects. These ModelAccessor obects allow for easier access
            to models

            ..code:: python
                # Load/save
                model = backend.models['key'].load()
                backend.models['key'].save(my_model_obj)

                # Predictions
                pred_train = backend.models['key'].predictions['train']
                pred_test = backend.models['key'].predictions['test']
        """
        return self._model_store

    @property
    def ensembles(self) -> EnsembleStore:
        """Get dictionary like access to ensembles stored

        Returns
        -------
        EnsembleStore
            A store of ensmbles that can be used in similar fashion to a
            dict where keys are ensemble names and values are EnsembleAccessor
            objects. These EnsembleAccessor obects allow for easier access
            to ensembles

            ..code:: python
                # Load/save
                ensemble = backend.ensembles['key'].load()
                backend.ensembles['key'].save(my_ensemble_obj)

                # Predictions
                pred_train = backend.ensembles['key'].predictions['train']
                pred_test = backend.ensembles['key'].predictions['test']
        """
        return self._ensembles_store

    def __del__(self) -> None:
        """Delete the folders if we do not retain them."""
        if not self.retain and self.context.exists(self.path):
            self.context.rmdir(self.path)
