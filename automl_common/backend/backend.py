from typing import Generic, Optional, TypeVar, Union

import tempfile
from pathlib import Path

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
            self.retain = retain if retain is not None else False
        elif isinstance(path, str):
            self.path = Path(path)
            self.retain = retain if retain is not None else True
        else:
            self.path = path
            self.retain = retain if retain is not None else True

        self.name = name

        self._model_store = ModelStore[ModelT](dir=self.model_dir)
        self._ensembles_store = EnsembleStore[ModelT](
            dir=self.ensemble_dir,
            model_dir=self.model_dir,
        )

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
        if not self.retain and self.path.exists():
            self.path.rmdir()
