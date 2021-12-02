from pathlib import Path

import pytest

import numpy as np

from automl_common.backend import Context, Ensemble

from .mocks import MockEnsemble


def test_construction(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to an existing tempdir

    context: Context
        A context to access the filesystem with

    Expects
    -------
    * The properties of the ensemble should be set
    * It should set exist to false as nothing has been saved yet
    """
    id = "hello"
    ensemble_folder = context.join(tmpdir, id)
    ensemble = Ensemble(id=id, dir=ensemble_folder, context=context)

    assert ensemble.id == id
    assert ensemble.dir == ensemble_folder
    assert ensemble.context == context
    assert not ensemble.exists


def test_construction_with_existing_dir(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to an existing tmpdir

    context: Context
        A context to access the filesystem with

    Expects
    -------
    * The properties of the ensemble should be set
    * It should `exist`
    """
    id = "hello"
    ensemble_dir = context.join(tmpdir, id)
    context.mkdir(ensemble_dir)

    ensemble = Ensemble(id=id, dir=ensemble_dir, context=context)
    assert ensemble.id == id
    assert ensemble.dir == ensemble_dir
    assert ensemble.context == context
    assert ensemble.exists


def test_ensemble_path(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble backend to test

    Expects
    -------
    * The ensemble path should be a fixed string from it's dir
    """
    expected = ensemble.context.join(ensemble.dir, "ensemble")
    assert ensemble.ensemble_path == expected


def test_save(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to test

    Expects
    -------
    * Save an ensemble creates a pickled object at ensemble_path
    * Should exist once saved
    """
    mock_ensemble = MockEnsemble(id=1)
    ensemble.save(mock_ensemble)

    assert ensemble.context.exists(ensemble.ensemble_path)
    assert ensemble.exists


def test_load_when_not_exists(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to test

    Expects
    -------
    * Should raise a RuntimeError if does not exist yet
    """
    assert not ensemble.exists
    with pytest.raises(RuntimeError):
        ensemble.load()


@pytest.mark.parametrize("id", [1,2,3])
def test_load_when_exists(ensemble: Ensemble, id: int):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to test

    id: int
        An id to give to the ensemble

    Expects
    -------
    * An ensemble should be able to be loaded form disk
    * The ensemble loaded should be exactly the same as the one that was saved
    """
    mock_ensemble = MockEnsemble(id=id)
    ensemble.save(mock_ensemble)

    assert ensemble.exists

    loaded = ensemble.load()
    assert loaded == mock_ensemble


def test_setup_on_clean_dir(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to test


    Expects
    -------
    * The setup should put the object in the state of existing
    * The setup should create its directory
    """
    ensemble.setup()

    assert ensemble.exists
    assert ensemble.context.exists(ensemble.dir)


def test_equality_with_self(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to check

    Expects
    -------
    * Should be equal to itself
    * Should not be != to itself
    """
    assert ensemble == ensemble
    assert ensemble.__eq__(ensemble)
    assert not ensemble != ensemble


def test_equality_with_other_equivalent_ensemble(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to check

    Expects
    -------
    * Should not equal a ensemble with a different id
    """
    other = Ensemble(id=ensemble.id, dir=ensemble.dir, context=ensemble.context)
    assert ensemble == other
    assert ensemble.__eq__(other)
    assert not ensemble != other


def test_equality_with_other_different_ensemble(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to check

    Expects
    -------
    * Should not equal an ensemble with a different id
    """
    other = Ensemble(id="nope", dir=ensemble.dir, context=ensemble.context)
    assert not ensemble == other
    assert not ensemble.__eq__(other)
    assert ensemble != other


def test_equality_works_with_str_conversion(ensemble: Ensemble):
    """
    Parameters
    ----------
    ensemble: Ensemble
        The ensemble to check

    Expects
    -------
    * Should equal the other ensemble if the only difference is a str conversion of the id
    """
    other = Ensemble(id=str(ensemble.id), dir=ensemble.dir, context=ensemble.context)
    assert ensemble == other
    assert ensemble.__eq__(other)
