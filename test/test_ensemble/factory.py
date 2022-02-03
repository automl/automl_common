from typing import Callable, Collection, Mapping, TypeVar

from pytest_cases import fixture

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import SingleEnsemble, UniformEnsemble, WeightedEnsemble
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble

MT = TypeVar("MT", bound=Model)


@fixture(scope="function")
def make_ensemble() -> Callable[..., MockEnsemble]:
    """Create mock ensemble objects

    ensemble = make_ensemble(model_dir=path, ids=[1,2,3])
    ensemble = make_ensemble(model_dir=path, ids={1: Model()})
    """

    def _make(model_store: ModelStore[MT], ids: Collection[str]) -> MockEnsemble:
        return MockEnsemble(model_store, ids=ids)

    return _make


@fixture(scope="function")
def make_weighted_ensemble() -> Callable[..., WeightedEnsemble[MT]]:
    """Create weighted ensemble objects

    ensemble = make_weighted_ensemble(model_dir=path, models={"a": 0.2, ... })
    ensemble = make_weighted_ensemble(model_dir=path, models={"a": (0.2, Model())})
    """

    def _make(
        model_store: ModelStore[MT],
        weighted_ids: Mapping[str, float],
    ) -> WeightedEnsemble[MT]:
        return WeightedEnsemble[MT](model_store, weighted_ids=weighted_ids)

    return _make


@fixture(scope="function")
def make_uniform_ensemble() -> Callable[..., UniformEnsemble[MT]]:
    """Create mock ensemble objects

    ensemble = make_uniform_ensemble(model_dir=path, models=["a", "b", "c"])
    ensemble = make_uniform_ensemble(model_dir=path, models={"a": Model()})
    """

    def _make(
        model_store: ModelStore[MT],
        ids: Collection[str],
    ) -> UniformEnsemble[MT]:
        return UniformEnsemble[MT](model_store, ids=ids)

    return _make


@fixture(scope="function")
def make_single_ensemble() -> Callable[..., SingleEnsemble[MT]]:
    """Create mock ensemble objects

    ensemble = make_single_ensemble(model_dir=path, model_id="a")
    ensemble = make_single_ensemble(model_dir=path, model_id="a", model=model)
    """

    def _make(
        model_store: ModelStore[MT],
        model_id: str,
    ) -> SingleEnsemble[MT]:
        return SingleEnsemble[MT](model_store, model_id)

    return _make
