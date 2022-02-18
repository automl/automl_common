"""
We have to create mocks because the checks will attempt to `fit` the ensemble where
as we rely on having fitted models before hand. To get around this, these mocks
will patch the `fit` of the ensemble to also fit some models in the ensemble store

Additionally, for classifiers, we have to ensure classes are encoded first. Normally
this should be handled by the individual models
"""
from __future__ import annotations

from typing import Any, Iterator, List, TypeVar
from typing_extensions import Literal

from itertools import product
from pathlib import Path

import numpy as np
from pytest_cases import parametrize
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


class MockSingleClassifierEnsemble(SingleClassifierEnsemble[CT]):
    def fit(self, x: np.ndarray, y: np.ndarray) -> MockSingleClassifierEnsemble[CT]:
        """Mock fit which will ensure models are fitted to the same data"""
        models = {str(i): DummyClassifier(strategy="stratified", random_state=0) for i in range(5)}
        for name, model in models.items():
            model.fit(x, y)
            self.model_store[name].save(model)

        return super().fit(x, y)


class MockWeightedClassifierEnsemble(WeightedClassifierEnsemble[CT]):
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> MockWeightedClassifierEnsemble[CT]:
        """Mock fit which will ensure models are fitted to the same data"""
        models = {str(i): DummyClassifier(strategy="stratified", random_state=0) for i in range(5)}
        for name, model in models.items():
            model.fit(x, y)
            self.model_store[name].save(model)

        return super().fit(x, y, *args, **kwargs)


def weighted_classifier_ensemble(path: Path) -> Iterator[MockWeightedClassifierEnsemble]:
    votings: List[Literal["majority", "probability"]] = ["probability", "majority"]
    sizes = [1, 3, 10]
    for voting, size in product(votings, sizes):

        dir = TMPDIR / f"weighted_classifier_ensemble_{voting}_{size}"
        dir.mkdir(parents=True)

        yield MockWeightedClassifierEnsemble[CT](
            model_store=ModelStore[CT](dir=dir), voting=voting, size=size
        )


def single_classifier_ensemble(path: Path) -> Iterator[MockSingleClassifierEnsemble]:
    name = "single_classifier_ensemble"
    dir = TMPDIR / name
    dir.mkdir(parents=True)

    yield MockSingleClassifierEnsemble[CT](model_store=ModelStore[CT](dir=dir))


def ensembles_to_test() -> Iterator[ClassifierEnsemble[CT]]:
    for generator in [weighted_classifier_ensemble, single_classifier_ensemble]:
        yield from generator(TMPDIR)


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
