from __future__ import annotations

from typing import TypeVar
from typing_extensions import (  # TODO: update with Python 3.8
    Protocol,
    runtime_checkable,
)

import numpy as np

from automl_common.model.model import Model

SelfT = TypeVar("SelfT")


@runtime_checkable
class Predictor(Model, Protocol):
    ...


@runtime_checkable
class Estimator(Predictor, Protocol):

    _estimator_type: str

    def fit(self: SelfT, x: np.ndarray, y: np.ndarray) -> SelfT:
        """Fit this estimator

        Parameters
        ----------
        x : np.ndarray
            The x data to fit to

        y : np.ndarray
            The y data to fit to

        Returns
        -------
        Estimator
            Self
        """
        ...


@runtime_checkable
class Regressor(Estimator, Predictor, Protocol):

    _estimator_type: str = "regressor"

    ...


@runtime_checkable
class Classifier(Estimator, Predictor, Protocol):

    _estimator_type: str = "classifier"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get the probability for the predictions

        Parameters
        ----------
        x : np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            The probability predictions
        """
        ...


@runtime_checkable
class TargetEncoder(Protocol):
    """An encoder that takes one argument ``y`` and produces classes_"""

    classes_: np.ndarray

    def fit(self: SelfT, y: np.ndarray, /) -> SelfT:
        """Fit the transformer

        Parameters
        ----------
        y : np.ndarray
            The array to transform

        Returns
        -------
        SelfT
        """
        ...

    def transform(self, y: np.ndarray, /) -> np.ndarray:
        """Transform the array

        Parameters
        ----------
        y : np.ndarray
            The array to transform

        Returns
        -------
        np.ndarray
            The transformed array
        """
        ...

    def fit_transform(self, y: np.ndarray, /) -> np.ndarray:
        """Fit and then transform the array

        Parameters
        ----------
        y : np.ndarray
            The array to transform

        Returns
        -------
        np.ndarray
            The transformed array
        """
        ...
