from typing import Callable, Mapping, Optional, Sequence, Union

from pathlib import Path

from pytest_cases import fixture

from automl_common.backend.stores import ModelStore
from automl_common.ensemble import SingleEnsemble, UniformEnsemble, WeightedEnsemble
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble


def store_models(path: Path, models: Mapping[str, Model]) -> None:
    """Stores models at path"""
    store = ModelStore[Model](path)
    for id, model in models.items():
        store[id].save(model)


@fixture(scope="function")
def make_ensemble() -> Callable[..., MockEnsemble]:
    """Create mock ensemble objects

    ensemble = make_ensemble(model_dir=path, ids=[1,2,3])
    ensemble = make_ensemble(model_dir=path, ids={1: Model()})
    """

    def _make(
        model_dir: Path,
        models: Union[Sequence[str], Mapping[str, Model]],
    ) -> MockEnsemble:
        if isinstance(models, Mapping):
            store_models(model_dir, models)
            ids = list(models.keys())

        else:
            ids = list(models)

        return MockEnsemble(model_dir, ids)

    return _make


@fixture(scope="function")
def make_weighted_ensemble() -> Callable[..., WeightedEnsemble[Model]]:
    """Create weighted ensemble objects

    ensemble = make_weighted_ensemble(model_dir=path, models={"a": 0.2, ... })
    ensemble = make_weighted_ensemble(model_dir=path, models={"a": (0.2, Model())})
    """

    def _make(
        model_dir: Path,
        weighted_ids: Mapping[str, float],
        models: Optional[Mapping[str, Model]] = None,
    ) -> WeightedEnsemble[Model]:
        if models is not None:
            store_models(model_dir, models)

        return WeightedEnsemble[Model](model_dir, weighted_identifiers=weighted_ids)

    return _make


@fixture(scope="function")
def make_uniform_ensemble() -> Callable[..., UniformEnsemble[Model]]:
    """Create mock ensemble objects

    ensemble = make_uniform_ensemble(model_dir=path, models=["a", "b", "c"])
    ensemble = make_uniform_ensemble(model_dir=path, models={"a": Model()})
    """

    def _make(
        model_dir: Path,
        models: Union[Sequence[str], Mapping[str, Model]],
    ) -> UniformEnsemble[Model]:
        if isinstance(models, Mapping):
            store_models(model_dir, models)
            models = list(models.keys())

        return UniformEnsemble[Model](model_dir, models)

    return _make


@fixture(scope="function")
def make_single_ensemble() -> Callable[..., SingleEnsemble[Model]]:
    """Create mock ensemble objects

    ensemble = make_single_ensemble(model_dir=path, model_id="a")
    ensemble = make_single_ensemble(model_dir=path, model_id="a", model=model)
    """

    def _make(
        model_dir: Path,
        model_id: str,
        model: Optional[Model] = None,
    ) -> SingleEnsemble[Model]:
        if model is not None:
            store_models(model_dir, {model_id: model})

        return SingleEnsemble[Model](model_dir, model_id)

    return _make
