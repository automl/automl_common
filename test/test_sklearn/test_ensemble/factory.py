from typing import Callable, Mapping, Optional, TypeVar, Union
from typing_extensions import Literal  # TODO, python 3.8

import numpy as np
from pytest_cases import fixture

from automl_common.backend.stores.model_store import ModelStore
from automl_common.sklearn.ensemble import (
    ClassifierEnsemble,
    Ensemble,
    RegressorEnsemble,
    SingleClassifierEnsemble,
    SingleEnsemble,
    SingleRegressorEnsemble,
    WeightedClassifierEnsemble,
    WeightedEnsemble,
    WeightedRegressorEnsemble,
)
from automl_common.sklearn.model import (
    Classifier,
    Predictor,
    ProbabilisticPredictor,
    Regressor,
)
from automl_common.util.types import Orderable

PredictorT = TypeVar("PredictorT", bound=Predictor)
ProbabilisticPredictorT = TypeVar(
    "ProbabilisticPredictorT", bound=ProbabilisticPredictor
)
ClassifierT = TypeVar("ClassifierT", bound=Classifier)
RegressorT = TypeVar("RegressorT", bound=Regressor)


@fixture(scope="function")
def make_sklearn_weighted_ensemble_predictors() -> Callable[
    ..., WeightedEnsemble[Predictor]
]:
    """Make a WeightedEnsemble[Predictor]

    Parameters
    ----------
    size: int = 10
        Size of the ensemble to build

    fitted: bool = False
        Whether to fit the model, if no x, y is provided, random data is used

    x: Optional[np.ndarray] = None,
        If provided, fits the ensemble to this x

    y: Optional[np.ndarray] = None,
        If provided, fits the ensemble to this y

    metric: Optional[Callable[[np.ndarray, np.ndarray], Orderable]] = rmse,
        The metric to use during fitting

    select: Literal["min", "max"] = "min",
        How to order the
    """

    def _make(
        size: int = 10,
        models: Union[int, Mapping[str, Predictor]] = 10,
        model_store: Optional[ModelStore[Predictor]] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        metric: Optional[Callable[[np.ndarray, np.ndarray], Orderable]] = None,
        select: Literal["min", "max"] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> WeightedEnsemble[Predictor]:
        raise NotImplementedError()

    return _make
