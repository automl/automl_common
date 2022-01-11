from typing import List, TypeVar

import numpy as np
from pytest_cases import fixture, parametrize

from automl_common.backend import Backend
from automl_common.ensemble import Ensemble
from automl_common.model import Model

from test.test_model.fixtures import MockModel

ModelT = TypeVar("ModelT", bound=Model)


class MockEnsemble(Ensemble):
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray
            The values to predict on

        Returns
        -------
        np.ndarray
            Just return what it got as input
        """
        return x


@fixture(scope="function")
def mock_ensemble(backend: Backend) -> MockEnsemble:
    """
    Parameters
    ----------
    backend: Backend
        The backend to use

    Returns
    -------
    MockEnsemble
        An ensemble with id "mock_ensemble" and three models "mock_model_{0,1,2}"
    """
    model_identifiers = [f"mock_model_{i}" for i in range(3)]
    ensemble_id = "mock_ensemble"

    for model_id in model_identifiers:
        mock_model = MockModel()
        backend.models[model_id].save(mock_model)

    mock_ensemble = MockEnsemble(backend=backend, identifiers=model_identifiers)
    backend.ensembles[ensemble_id].save(mock_ensemble)

    return mock_ensemble


@fixture(scope="function")
def mock_ensembles(backend: Backend) -> List[MockEnsemble]:
    """
    Parameters
    ----------
    backend: Backend
        The backend to use

    Returns
    -------
    List[MockEnsemble]
        Ensembles with id "mock_ensemble_{0,1,2},
        each with three models "mock_model_{0,1,2}"
    """
    model_identifiers = [f"mock_ensemble_{i}" for i in range(3)]

    for model_id in model_identifiers:
        mock_model = MockModel()
        backend.models[model_id].save(mock_model)

    mock_ensembles = {
        str(i): MockEnsemble(backend=backend, identifiers=model_identifiers)
        for i in range(3)
    }

    for id, mock_ensemble in mock_ensembles.items():
        backend.ensembles[id].save(mock_ensemble)

    return list(mock_ensembles.values())


@fixture(scope="function")
@parametrize("impl", [mock_ensemble])
def ensemble(impl: Ensemble) -> Ensemble:
    """Forwards Ensemble objects"""
    return impl


@fixture(scope="function")
@parametrize("impl", [mock_ensembles])
def ensembles(impl: List[Ensemble]) -> List[Ensemble]:
    """Forwards different Lists of Ensembles"""
    return impl
