from typing_extensions import (  # TODO: update with Python 3.8
    Protocol,
    runtime_checkable,
)

from automl_common.model.model import Model, ProbabilisticModel


@runtime_checkable
class Predictor(Model, Protocol):
    ...


@runtime_checkable
class ProbabilisticPredictor(Predictor, ProbabilisticModel, Protocol):
    ...


@runtime_checkable
class Regressor(Predictor, Protocol):
    ...


@runtime_checkable
class Classifier(ProbabilisticPredictor, Protocol):
    ...
