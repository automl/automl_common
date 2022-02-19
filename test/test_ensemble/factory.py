from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pathlib import Path

from pytest_cases import fixture

from automl_common.backend.stores.model_store import ModelStore
from automl_common.ensemble import SingleEnsemble, UniformEnsemble, WeightedEnsemble
from automl_common.model import Model

from test.test_ensemble.mocks import MockEnsemble

MT = TypeVar("MT", bound=Model)


def setup_models(
    models: Union[int, Iterable[str], Mapping[str, MT]],
    model_type: Optional[Type[MT]] = None,
    model_store: Optional[ModelStore[MT]] = None,
    path: Optional[Path] = None,
) -> Tuple[ModelStore[MT], List[str]]:
    """Create a model store with models in it

    Parameters
    ----------
    models : Union[int, Iterable[str], Mapping[str, MT]]
        The models to put into the store. If not providing direct models,
        must provide the type.

    model_type : Optional[Type[MT]] = None
        The type of model to build if no instances provided

    model_store : Optional[ModelStore[MT]] = None
        The store to place models into. Must provide path if not provided.

    path : Optional[Path] = None
        The path to make the model store at

    Returns
    -------
    ModelStore[MT], List[str]
        The model store with models placed in it along with the list
        of model ids for the ensemble
    """
    if path is None:
        assert model_store is not None
        store = model_store

    elif model_store is None:
        assert path is not None
        store = ModelStore[MT](dir=path)

    else:
        raise NotImplementedError()

    if isinstance(models, int):
        assert model_type is not None
        models = {str(i): model_type() for i in range(models)}

    elif isinstance(models, Iterable):
        assert model_type is not None
        models = {key: model_type() for key in models}

    elif isinstance(models, Mapping):  # pragma: no cover
        models = models

    else:
        raise NotImplementedError()

    assert len(models) > 0

    for key, model in models.items():
        store[key].save(model)

    ids = list(models)

    return store, ids


@fixture(scope="function")
def make_ensemble() -> Callable[..., MockEnsemble[MT]]:
    """Create mock ensemble objects"""

    def _make(
        models: Union[int, Iterable[str], Mapping[str, MT]],
        model_type: Optional[Type[MT]] = None,
        model_store: Optional[ModelStore[MT]] = None,
        path: Optional[Path] = None,
    ) -> MockEnsemble:
        store, ids = setup_models(
            models=models,
            model_type=model_type,
            model_store=model_store,
            path=path,
        )

        return MockEnsemble[MT](store, ids=ids)

    return _make


@fixture(scope="function")
def make_weighted_ensemble() -> Callable[..., WeightedEnsemble[MT]]:
    """Create weighted ensemble objects

    ensemble = make_weighted_ensemble(model_dir=path, models={"a": 0.2, ... })
    ensemble = make_weighted_ensemble(model_dir=path, models={"a": (0.2, Model())})
    """

    def _make(
        models: Union[int, Iterable[str], Mapping[str, MT]],
        weights: Union[Iterable[float], Mapping[str, float], None] = None,
        model_type: Optional[Type[MT]] = None,
        model_store: Optional[ModelStore[MT]] = None,
        path: Optional[Path] = None,
    ) -> WeightedEnsemble[MT]:
        store, ids = setup_models(
            models=models,
            model_type=model_type,
            model_store=model_store,
            path=path,
        )

        if weights is None:
            weighted_ids = {id: 1.0 for id in ids}

        elif isinstance(weights, Mapping):
            weighted_ids = dict(weights)

        elif isinstance(weights, Iterable) and not isinstance(weights, Mapping):
            weighted_ids = {id: weight for id, weight in zip(ids, weights)}
            assert set(ids) == set(weighted_ids)

        else:
            raise NotImplementedError()

        return WeightedEnsemble[MT](store, weighted_ids=weighted_ids)

    return _make


@fixture(scope="function")
def make_uniform_ensemble() -> Callable[..., UniformEnsemble[MT]]:
    """Create mock ensemble objects

    ensemble = make_uniform_ensemble(model_dir=path, models=["a", "b", "c"])
    ensemble = make_uniform_ensemble(model_dir=path, models={"a": Model()})
    """

    def _make(
        models: Union[int, Iterable[str], Mapping[str, MT]],
        model_type: Optional[Type[MT]] = None,
        model_store: Optional[ModelStore[MT]] = None,
        path: Optional[Path] = None,
    ) -> UniformEnsemble[MT]:
        store, ids = setup_models(
            models=models,
            model_type=model_type,
            model_store=model_store,
            path=path,
        )

        return UniformEnsemble[MT](store, ids=ids)

    return _make


@fixture(scope="function")
def make_single_ensemble() -> Callable[..., SingleEnsemble[MT]]:
    """Create mock ensemble objects

    ensemble = make_single_ensemble(model_dir=path, model_id="a")
    ensemble = make_single_ensemble(model_dir=path, model_id="a", model=model)
    """

    def _make(
        model: Union[str, Tuple[str, MT], None] = None,
        model_type: Optional[Type[MT]] = None,
        model_store: Optional[ModelStore[MT]] = None,
        path: Optional[Path] = None,
    ) -> SingleEnsemble[MT]:

        models: Union[List[str], Dict[str, MT]]
        if model is None:
            models = ["model_id"]

        elif isinstance(model, str):
            models = [model]

        elif isinstance(model, tuple):
            models = dict([model])

        store, ids = setup_models(
            models=models,
            model_type=model_type,
            model_store=model_store,
            path=path,
        )
        return SingleEnsemble[MT](store, ids[0])

    return _make
