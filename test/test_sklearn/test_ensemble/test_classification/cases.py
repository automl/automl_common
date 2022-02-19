from typing import Callable, List, Tuple, Union

from pathlib import Path

from pytest_cases import case, parametrize

import numpy as np

from automl_common.sklearn.ensemble.classification import (
    SingleClassifierEnsemble,
    WeightedClassifierEnsemble,
)

from test.test_sklearn.test_models.mocks import MockClassifier

SCE = SingleClassifierEnsemble[MockClassifier]
WCE = WeightedClassifierEnsemble[MockClassifier]


@case(tags=["single", "classifier"])
def case_single_classifier(
    path: Path,
    make_sklearn_single_classifier_ensemble: Callable[..., SCE],
) -> SCE:
    """A SingleClassifierEnsemble not fitted"""
    return make_sklearn_single_classifier_ensemble(path=path, fitted=False)


@case(tags=["single", "classifier", "fitted"])
@parametrize("classes", [1, 2, 3, [[0, 0, 0], [0, 1, 1], [1, 1, 1]]])
def case_single_classifier_fitted(
    path: Path,
    classes: Union[int, List[int]],
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    make_sklearn_single_classifier_ensemble: Callable[..., SCE],
) -> SCE:
    """A SingleClassifierEnsemble fitted"""
    x, y = make_xy("classification", classes=classes)
    return make_sklearn_single_classifier_ensemble(
        path=path,
        fitted=True,
        x=x,
        y=y,
        model_x=x,
        model_y=y,
    )


@case(tags=["weighted", "classifier"])
def case_weighted_classifier(
    path: Path,
    make_sklearn_weighted_classifier_ensemble: Callable[..., WCE],
) -> WCE:
    """A WeightedRegressorEnsemble not fitted"""
    return make_sklearn_weighted_classifier_ensemble(path=path, fitted=False)


@case(tags=["weighted", "classifier", "fitted"])
@parametrize("voting", ["probability", "majority"])
@parametrize("classes", [1, 2, 3, [[0, 0, 0], [0, 1, 1], [1, 1, 1]]])
def case_weighted_classifier_fitted(
    path: Path,
    voting: str,
    make_xy: Callable[..., Tuple[np.ndarray, np.ndarray]],
    classes: Union[int, List[int]],
    make_sklearn_weighted_classifier_ensemble: Callable[..., WCE],
) -> WCE:
    """A WeightedRegressorEnsemble fitted"""
    x, y = make_xy("classification", classes=classes)
    return make_sklearn_weighted_classifier_ensemble(
        path=path,
        voting=voting,
        fitted=True,
        x=x,
        y=y,
        model_x=x,
        model_y=y,
    )
