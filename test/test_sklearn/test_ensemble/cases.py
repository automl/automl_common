from typing import Callable, List, Tuple

from pathlib import Path

import numpy as np
from pytest_cases import case, parametrize

from automl_common.sklearn.ensemble import (
    SingleClassifierEnsemble,
    SingleRegressorEnsemble,
    WeightedClassifierEnsemble,
    WeightedRegressorEnsemble,
)

from test.test_sklearn.test_models.mocks import MockClassifier, MockRegressor

SRE = SingleRegressorEnsemble[MockRegressor]
SCE = SingleClassifierEnsemble[MockClassifier]
WRE = WeightedRegressorEnsemble[MockRegressor]
WCE = WeightedClassifierEnsemble[MockClassifier]


@case(tags=["single", "regressor"])
def case_single_regressor(
    path: Path,
    make_sklearn_single_regressor_ensemble: Callable[..., SRE],
) -> SRE:
    """A SingleRegressorEnsemble not fitted"""
    return make_sklearn_single_regressor_ensemble(path=path, fitted=False)


@case(tags=["single", "classifier"])
def case_single_classifier(
    path: Path,
    make_sklearn_single_classifier_ensemble: Callable[..., SCE],
) -> SCE:
    """A SingleClassifierEnsemble not fitted"""
    return make_sklearn_single_classifier_ensemble(path=path, fitted=False)


@case(tags=["single", "regressor", "fitted"])
def case_single_regressor_fitted(
    path: Path,
    make_sklearn_single_regressor_ensemble: Callable[..., SRE],
) -> SRE:
    """A SingleRegressorEnsemble not fitted"""
    return make_sklearn_single_regressor_ensemble(path=path, fitted=True)


@case(tags=["single", "classifier", "fitted"])
def case_single_classifier_fitted(
    path: Path,
    make_sklearn_single_classifier_ensemble: Callable[..., SCE],
) -> SCE:
    """A SingleClassifierEnsemble fitted"""
    return make_sklearn_single_classifier_ensemble(path=path, fitted=True)


@case(tags=["weighted", "regressor"])
def case_weighted_regressor(
    path: Path,
    make_sklearn_weighted_regressor_ensemble: Callable[..., WRE],
) -> WRE:
    """A WeightedRegressorEnsemble not fitted"""
    return make_sklearn_weighted_regressor_ensemble(path=path, fitted=False)


@case(tags=["weighted", "classifier"])
def case_weighted_classifier(
    path: Path,
    make_sklearn_weighted_classifier_ensemble: Callable[..., WCE],
) -> WCE:
    """A WeightedRegressorEnsemble not fitted"""
    return make_sklearn_weighted_classifier_ensemble(path=path, fitted=False)


@case(tags=["weighted", "regressor", "fitted"])
@parametrize("targets", [1, 3])
def case_weighted_regressor_fitted(
    path: Path,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    targets: int,
    make_sklearn_weighted_regressor_ensemble: Callable[..., WRE],
) -> WRE:
    """A WeightedRegressorEnsemble fitted"""
    x, y = make_xy(kind="regression", targets=targets)
    return make_sklearn_weighted_regressor_ensemble(
        path=path, fitted=True, model_x=x, model_y=y, x=y, y=y
    )


@case(tags=["weighted", "classifier", "fitted"])
@parametrize("voting", ["majority", "probability"])
@parametrize("classes", [[0, 1], [[1, 0, 1], [1, 0, 1]]])
def case_weighted_classifier_fitted(
    path: Path,
    voting: str,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    classes: List,
    make_sklearn_weighted_classifier_ensemble: Callable[..., WCE],
) -> WCE:
    """A WeightedRegressorEnsemble fitted"""
    x, y = make_xy("classification", classes=classes)
    return make_sklearn_weighted_classifier_ensemble(path=path, voting=voting, fitted=True)
