"""
We have to create mocks because the checks will attempt to `fit` the ensemble where
as we rely on having fitted models before hand. To get around this, these mocks
will patch the `fit` of the ensemble to also fit some models in the ensemble store

Additionally, for classifiers, we have to ensure classes are encoded first. Normally
this should be handled by the individual models
"""
from __future__ import annotations

from typing import Iterator, List, Optional, TypeVar
from typing_extensions import Literal

from itertools import product
from pathlib import Path

import pytest
from pytest_cases import parametrize

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.utils.estimator_checks import check_estimator

from automl_common.backend.stores.model_store import ModelStore
from automl_common.sklearn.ensemble import (
    ClassifierEnsemble,
    SingleClassifierEnsemble,
    WeightedClassifierEnsemble,
)
from automl_common.sklearn.model import Classifier

from test.conftest import manual_tmp

TMPDIR = manual_tmp / "sklearn_ensemble_classifiers"

CT = TypeVar("CT", bound=Classifier)
ID = TypeVar("ID")


class MockSingleClassifierEnsemble(SingleClassifierEnsemble[str, CT]):
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pred_key: Optional[str] = None,
    ) -> MockSingleClassifierEnsemble[CT]:
        """Mock fit which will ensure models are fitted to the same data"""
        models = {str(i): DummyClassifier(strategy="stratified", random_state=0) for i in range(5)}
        for name, model in models.items():
            model.fit(x, y)
            self.model_store[name].save(model)

        return super().fit(x, y, pred_key)


class MockWeightedClassifierEnsemble(WeightedClassifierEnsemble[str, CT]):
    def fit(
        self, x: np.ndarray, y: np.ndarray, pred_key: Optional[str] = None
    ) -> MockWeightedClassifierEnsemble[CT]:
        """Mock fit which will ensure models are fitted to the same data"""
        models = {str(i): DummyClassifier(strategy="stratified", random_state=0) for i in range(5)}
        for name, model in models.items():
            model.fit(x, y)
            self.model_store[name].save(model)

        return super().fit(x, y, pred_key)


def weighted_classifier_ensemble(path: Path) -> Iterator[MockWeightedClassifierEnsemble]:
    votings: List[Literal["majority", "probability"]] = ["probability", "majority"]
    sizes = [1, 3, 10]
    for voting, size in product(votings, sizes):

        dir = TMPDIR / f"weighted_classifier_ensemble_{voting}_{size}"
        dir.mkdir(parents=True)

        yield MockWeightedClassifierEnsemble[CT](
            model_store=ModelStore(dir=dir),
            voting=voting,
            size=size,
        )


def single_classifier_ensemble(path: Path) -> Iterator[MockSingleClassifierEnsemble]:
    name = "single_classifier_ensemble"
    dir = TMPDIR / name
    dir.mkdir(parents=True)

    yield MockSingleClassifierEnsemble[CT](model_store=ModelStore(dir=dir))


def ensembles_to_test() -> Iterator[ClassifierEnsemble[ID, CT]]:
    for generator in [weighted_classifier_ensemble, single_classifier_ensemble]:
        yield from generator(TMPDIR)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.SkipTestWarning")
@pytest.mark.filterwarnings("ignore:Expensive function `_more_tags` called")
@pytest.mark.filterwarnings("ignore:Can't check dok sparse matrix for nan or inf.")
@pytest.mark.sklearn
@parametrize("ensemble", list(ensembles_to_test()))
def test_compatibility(ensemble: ClassifierEnsemble) -> None:
    check_estimator(ensemble)


"""
@parametrize_with_checks(list(ensembles_to_test()))
def test_sklearn_compatibility(
    check: Callable,
    estimator: Callable,
) -> None:
    check(estimator)
"""
