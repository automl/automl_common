from typing import Callable

from pathlib import Path

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.backend.accessors import EnsembleAccessor
from automl_common.ensemble import Ensemble
from automl_common.model import Model

import test.test_backend.test_accessors.cases as cases


def test_construction(path: Path) -> None:
    """
    Parameters
    ----------
    path: Path
        The base path

    Expects
    -------
    * Should construct without issue
    * It's _ids should remain None at construction
    """
    dir = path / "ensemble"
    model_dir = path / "models"
    accessor = EnsembleAccessor[Model](dir, model_dir)
    assert accessor._ids is None


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ~ft.has_tag("populated")
)
def test_ids_unpopulated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        An unpopulated EnsembleAccessor

    Expects
    -------
    * Should give empty ids if not populated
    """
    assert accessor.ids == []


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ft.has_tag("populated")
)
def test_ids_populated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        A populated EnsembleAccessor

    Expects
    -------
    * Should be non-empty
    * Should contain the ids in iterator
    """
    assert len(accessor.ids) > 0
    assert list(accessor.ids) == list(iter(accessor))


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ~ft.has_tag("populated")
)
def test_models_unpopulated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        An unpopulated EnsembleAccessor

    Expects
    -------
    * Should raise a RuntimeError as we do not know what models are contained
    """
    with pytest.raises(RuntimeError):
        accessor.models


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ft.has_tag("populated")
)
def test_models_populated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        A populated EnsembleAccessor

    Expects
    -------
    * Should return a FilteredModelStore with the id's in the ensemble
    """
    assert list(accessor.models) == accessor.ids


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ft.has_tag("populated")
)
def test_iter_populated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        A populated EnsembleAccessor

    Expects
    -------
    * Should be a non-empty iterator
    * Should contain the same ids as the EnsembleAccessor
    """
    assert len(list(accessor)) > 0
    assert list(accessor) == accessor.ids


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ~ft.has_tag("populated")
)
def test_iter_unpopulated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        An unpopulated EnsembleAccessor

    Expects
    -------
    * The iter should be empty
    """
    assert list(iter(accessor)) == []


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ft.has_tag("populated")
)
def test_contains(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor
        A populated EnsembleAccessor

    Expects
    -------
    * Should correctly contain a model that is in it item in it
    """
    model = next(iter(accessor))
    assert model in accessor


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ~ft.has_tag("populated")
)
def test_contains_badkey(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor

    Expects
    -------
    * Should not contain a key that is not present in the Ensemble
    """
    badkey = "this_is_a_badkey"
    assert badkey not in accessor


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ft.has_tag("populated")
)
def test_getitem(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor

    Expects
    -------
    * Should be able to retrieve every model it contains
    """
    models = list(iter(accessor))
    assert len(models) > 0

    assert all(accessor[model] is not None for model in models)


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ~ft.has_tag("populated")
)
def test_getitem_badkey(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor

    Expects
    -------
    * Should raise a KeyError if attempting to retrieve a model which is not contained
    """
    bad_model_key = "this_is_a_bad_model_key"
    assert bad_model_key not in accessor

    with pytest.raises(KeyError):
        accessor[bad_model_key]


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ft.has_tag("populated")
)
def test_len_populated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor

    Expects
    -------
    * The length of the ensemble should be > 0 if populated
    """
    assert list(iter(accessor)) != []  # Validty of accessor recieved being populated

    assert len(accessor) > 0


@parametrize_with_cases(
    "accessor", cases=cases, filter=ft.has_tag("ensemble") & ~ft.has_tag("populated")
)
def test_len_unpopulated(accessor: EnsembleAccessor) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor

    Expects
    -------
    * The length of an unpopulated EnsembleAccessor is 0
    """
    assert list(iter(accessor)) == []  # Validty of accessor recieved being unpopulated

    assert len(accessor) == 0


def test_eq_same(
    path: Path,
    make_model: Callable[..., Model],
    make_ensemble: Callable[..., Ensemble],
    make_ensemble_accessor: Callable[..., EnsembleAccessor],
) -> None:
    """
    Parameters
    ----------
    path: Path
        A path to build from

    make_model: Callable[..., Model],
        A factory for models

    make_ensemble: Callable[..., Ensemble],
        A factory for ensembles

    make_ensemble_accessor: Callable[..., EnsembleAccessor]
        A factory for an EnsembleAccessor

    Expects
    -------
    * An EnsembleAccessor constructed with the same params should be equal
    """
    ensemble_dir = path / "ensemble"
    model_dir = path / "models"
    models = {k: make_model() for k in "abc"}
    ensemble = make_ensemble(model_dir, models)

    ensemble_accessor_1 = make_ensemble_accessor(ensemble_dir, model_dir, ensemble)
    ensemble_accessor_2 = make_ensemble_accessor(ensemble_dir, model_dir, ensemble)

    assert ensemble_accessor_1 == ensemble_accessor_2


def test_eq_different(
    path: Path,
    make_model: Callable[..., Model],
    make_ensemble: Callable[..., Ensemble],
    make_ensemble_accessor: Callable[..., EnsembleAccessor],
) -> None:
    """
    Parameters
    ----------
    accessor: EnsembleAccessor

    TODO
        There is defnitely more cases that could be tested here

    Expects
    -------
    * An ensemble accessor in a different dir should not be equal
    """
    ensemble_dir_1 = path / "ensemble_1"
    ensemble_dir_2 = path / "ensemble_2"

    model_dir = path / "models"

    ensemble_accessor_1 = make_ensemble_accessor(ensemble_dir_1, model_dir)
    ensemble_accessor_2 = make_ensemble_accessor(ensemble_dir_2, model_dir)

    assert ensemble_accessor_1 != ensemble_accessor_2


@parametrize_with_cases("accessor", cases=cases, filter=ft.has_tag("predictions"))
def test_predictions_with_predictions(accessor: EnsembleAccessor) -> None:
    """
    accessor: Accessor
        An EnsembleAccessor with predictions stored

    Expects
    -------
    * Should have a non 0 length
    * Should contain all the predictions it iterates through
    """
    assert len(accessor.predictions) > 0
    for key in accessor.predictions:
        assert accessor.predictions[key] is not None


@parametrize_with_cases("accessor", cases=cases, filter=~ft.has_tag("predictions"))
def test_predictions_no_predictions(accessor: EnsembleAccessor) -> None:
    """
    accessor: EnsembleAccessor
        An EnsembleAccessor with no predictions stored

    Expects
    -------
    * Should have a 0 length
    * Should provide an empty iterator
    """
    assert len(accessor.predictions) == 0
    assert len(list(accessor.predictions)) == 0
