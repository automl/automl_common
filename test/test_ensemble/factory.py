from typing import Callable, Mapping, Optional, Sequence, TypeVar, Union

from pathlib import Path

from pytest_cases import fixture

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import SingleEnsemble, UniformEnsemble, WeightedEnsemble
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble

MT = TypeVar("MT", bound=Model)


def store_models(path: Path, models: Mapping[str, MT]) -> None:
    """Stores models at path"""
    store = ModelStore[MT](path)
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
        models: Union[Sequence[str], Mapping[str, MT]],
    ) -> MockEnsemble:
        if isinstance(models, Mapping):
            store_models(model_dir, models)
            ids = list(models.keys())

        else:
            ids = list(models)

        return MockEnsemble(model_dir, ids)

    return _make


@fixture(scope="function")
def make_weighted_ensemble() -> Callable[..., WeightedEnsemble[MT]]:
    """Create weighted ensemble objects

    ensemble = make_weighted_ensemble(model_dir=path, models={"a": 0.2, ... })
    ensemble = make_weighted_ensemble(model_dir=path, models={"a": (0.2, Model())})
    """

    def _make(
        model_dir: Path,
        weighted_ids: Mapping[str, float],
        models: Optional[Mapping[str, MT]] = None,
    ) -> WeightedEnsemble[MT]:
        if models is not None:
            store_models(model_dir, models)

        return WeightedEnsemble[MT](model_dir, weighted_ids=weighted_ids)

    return _make


@fixture(scope="function")
def make_uniform_ensemble() -> Callable[..., UniformEnsemble[MT]]:
    """Create mock ensemble objects

    ensemble = make_uniform_ensemble(model_dir=path, models=["a", "b", "c"])
    ensemble = make_uniform_ensemble(model_dir=path, models={"a": Model()})
    """

    def _make(
        model_dir: Path,
        models: Union[Sequence[str], Mapping[str, MT]],
    ) -> UniformEnsemble[MT]:
        if isinstance(models, Mapping):
            store_models(model_dir, models)
            models = list(models.keys())

        return UniformEnsemble[MT](model_dir, models)

    return _make


@fixture(scope="function")
def make_single_ensemble() -> Callable[..., SingleEnsemble[MT]]:
    """Create mock ensemble objects

    ensemble = make_single_ensemble(model_dir=path, model_id="a")
    ensemble = make_single_ensemble(model_dir=path, model_id="a", model=model)
    """

    def _make(
        model_dir: Path,
        model_id: str,
        model: Optional[MT] = None,
    ) -> SingleEnsemble[MT]:
        if model is not None:
            store_models(model_dir, {model_id: model})

        return SingleEnsemble[MT](model_dir, model_id)

    return _make
