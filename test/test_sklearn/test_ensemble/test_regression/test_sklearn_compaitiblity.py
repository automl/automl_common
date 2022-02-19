"""
We have to create mocks because the checks will attempt to `fit` the ensemble where
as we rely on having fitted models before hand. To get around this, these mocks
will patch the `fit` of the ensemble to also fit some models in the ensemble store
"""
from __future__ import annotations

from typing import Any, Iterator, TypeVar

from pathlib import Path

import pytest
from pytest_cases import parametrize

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.utils.estimator_checks import check_estimator

from automl_common.backend.stores.model_store import ModelStore
from automl_common.sklearn.ensemble import (
    RegressorEnsemble,
    SingleRegressorEnsemble,
    WeightedRegressorEnsemble,
)
from automl_common.sklearn.model import Regressor

from test.conftest import manual_tmp

TMPDIR = manual_tmp / "ensemble_sklearn_compatibility"

RT = TypeVar("RT", bound=Regressor)


class MockWeightedRegressorEnsemble(WeightedRegressorEnsemble[RT]):
    def fit(
        self, x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any
    ) -> MockWeightedRegressorEnsemble[RT]:
        """Mock fit which will ensure models are fitted to the same data"""
        models = {str(i): DummyRegressor() for i in range(5)}
        for name, model in models.items():
            model.fit(x, y)
            self.model_store[name].save(model)

        return super().fit(x, y)  # type: ignore


class MockSingleRegressorEnsemble(SingleRegressorEnsemble[RT]):
    def fit(self, x: np.ndarray, y: np.ndarray) -> MockSingleRegressorEnsemble[RT]:
        """Mock fit which will ensure models are fitted to the same data"""
        models = {str(i): DummyRegressor() for i in range(5)}
        for name, model in models.items():
            model.fit(x, y)
            self.model_store[name].save(model)

        return super().fit(x, y)  # type: ignore


def weighted_regressor_ensemble(path: Path) -> Iterator[MockWeightedRegressorEnsemble]:
    dir = TMPDIR / "weighted_regressor_ensemble"
    dir.mkdir(parents=True)

    yield MockWeightedRegressorEnsemble[RT](model_store=ModelStore[RT](dir=dir))


def single_regressor_ensemble(path: Path) -> Iterator[MockSingleRegressorEnsemble]:
    name = "single_classifier_ensemble"
    dir = TMPDIR / name
    dir.mkdir(parents=True)

    yield MockSingleRegressorEnsemble[RT](model_store=ModelStore[RT](dir=dir))


def ensembles_to_test() -> Iterator[RegressorEnsemble[RT]]:
    for generator in [single_regressor_ensemble, weighted_regressor_ensemble]:
        yield from generator(TMPDIR)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.SkipTestWarning")
@pytest.mark.filterwarnings("ignore:Expensive function `_more_tags` called")
@pytest.mark.filterwarnings("ignore:Can't check dok sparse matrix for nan or inf.")
@parametrize("ensemble", list(ensembles_to_test()))
def test_compatibility(ensemble: RegressorEnsemble) -> None:
    check_estimator(ensemble)


"""
@parametrize_with_checks(list(ensembles_to_test()))
def test_sklearn_compatibility(
    check: Callable,
    estimator: Callable,
) -> None:
    check(estimator)
"""
