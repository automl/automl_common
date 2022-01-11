from typing import List

import numpy as np
from pytest_cases import fixture, parametrize

from automl_common.model import Model


class MockModel(Model):
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
def mock_model() -> MockModel:
    """
    Returns
    -------
    MockModel
        A mock model
    """
    return MockModel()


@fixture(scope="function")
def mock_models() -> List[MockModel]:
    """
    Returns
    -------
    MockModel
        A list of 3 mock models
    """
    return [MockModel() for _ in range(3)]


@fixture(scope="function")
@parametrize("impl", [mock_model])
def model(impl: Model) -> Model:
    """A model"""
    return impl


@fixture(scope="function")
@parametrize("impl", [mock_models])
def models(impl: List[Model]) -> List[Model]:
    """A list of models"""
    return impl
