from abc import ABC

from pathlib import Path

from automl_common.backend.accessors import ModelAccessor
from automl_common.ensemble import Ensemble


# Mostly just uses functionlality of ModelAccessor
class EnsembleAccessor(ABC, ModelAccessor[Ensemble]):
    """The state of an Ensemble with a directory on a filesystem.

    As automl_common manages ensembling in general, we can keep
    these as picklable items and hence implement what it means
    to load and save an ensemble.

    Manages a directory:
    /<path>
        / predictions_train.npy
        / predictions_test.npy
        / predictions_val.npy
        / ensemble
        / ...

    Any implementing class can add more state that can be managed about this model.

    An EnsembleView must implement:
    * `save` - Save a model to a backend
    * `load` - Load a model from a backend
    """

    @property
    def path(self) -> Path:
        """Path to the ensemble object"""
        return self.path / "ensemble"
