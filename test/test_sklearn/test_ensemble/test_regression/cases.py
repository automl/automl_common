from typing import Callable, Tuple, TypeVar

from pathlib import Path

from pytest_cases import case, parametrize

import numpy as np

from automl_common.sklearn.ensemble.regression import (
    SingleRegressorEnsemble,
    WeightedRegressorEnsemble,
)

from test.test_sklearn.test_models.mocks import MockRegressor

ID = TypeVar("ID")

SRE = SingleRegressorEnsemble[ID, MockRegressor]
WRE = WeightedRegressorEnsemble[ID, MockRegressor]


@case(tags=["single", "regressor"])
def case_single_regressor(
    path: Path,
    make_sklearn_single_regressor_ensemble: Callable[..., SRE],
) -> SRE:
    """A SingleRegressorEnsemble not fitted"""
    return make_sklearn_single_regressor_ensemble(path=path, fitted=False)


@case(tags=["single", "regressor", "fitted"])
@parametrize("targets", [1, 2, 3])
def case_single_regressor_fitted(
    path: Path,
    targets: int,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    make_sklearn_single_regressor_ensemble: Callable[..., SRE],
) -> SRE:
    """A SingleRegressorEnsemble not fitted"""
    x, y = make_xy("classification", targets=targets)
    return make_sklearn_single_regressor_ensemble(
        path=path,
        fitted=True,
        x=x,
        y=y,
        model_x=x,
        model_y=y,
    )


@case(tags=["weighted", "regressor"])
def case_weighted_regressor(
    path: Path,
    make_sklearn_weighted_regressor_ensemble: Callable[..., WRE],
) -> WRE:
    """A WeightedRegressorEnsemble not fitted"""
    return make_sklearn_weighted_regressor_ensemble(path=path, fitted=False)


@case(tags=["weighted", "regressor", "fitted"])
@parametrize("targets", [1, 2, 3])
def case_weighted_regressor_fitted(
    path: Path,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    targets: int,
    make_sklearn_weighted_regressor_ensemble: Callable[..., WRE],
) -> WRE:
    """A WeightedRegressorEnsemble fitted"""
    x, y = make_xy(kind="regression", targets=targets)
    return make_sklearn_weighted_regressor_ensemble(
        path=path,
        fitted=True,
        x=x,
        y=y,
        model_x=x,
        model_y=y,
    )
