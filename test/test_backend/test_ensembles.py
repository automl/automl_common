import pytest

from pathlib import Path

import numpy as np

from automl_common.backend import Ensembles, Context

def test_construction(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path that exists

    context: Context
        A context to interact with the filesystem

    Expects
    -------
    * Should construct without errors and set attributes
    """
    ensembles = Ensembles(dir=tmpdir, context=context)
    assert ensembles.dir == tmpdir
    assert ensembles.context == context

@pytest.mark.parametrize("ensembles", [[1,2,3]], indirect=True)
def test_len_non_empty(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The ensembles backend object to test

    Expects
    -------
    * Should have the same lenegth as the number of ensembles it contains
    """
    assert len(ensembles) == 3


@pytest.mark.parametrize("ensembles", [[]], indirect=True)
def test_len_empty(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should have no length
    """
    assert len(ensembles) == 0


@pytest.mark.parametrize("ensembles", [[1, 2, 3]], indirect=True)
def test_contains_with_ensemble_contained_int(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should have the ensemble with the id contained
    """
    assert all(i in ensembles for i in [1, 2, 3])
    assert all(str(i) in ensembles for i in [1, 2, 3])


@pytest.mark.parametrize("ensembles", [[1, 2, 3]], indirect=True)
def test_contains_without_contained_ensemble(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should give false for containing a ensemble it does not contain
    """
    assert 4 not in ensembles
    assert str(4) not in ensembles


@pytest.mark.parametrize("ensembles", [[(1, 1), (2, 2)]], indirect=True)
def test_contains_with_ensemble_contained_tuple(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should have the ensemble with the id contained
    """
    assert (1, 1) in ensembles and (2, 2) in ensembles
    assert str((1, 1)) in ensembles and str((2, 2)) in ensembles


@pytest.mark.parametrize("ensembles", [[(1, 1), (2, 2)]], indirect=True)
def test_contains_without_contain_ensemble_tuple(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should give false for ensemble it does not contain
    """
    assert (3, 3) not in ensembles
    assert str((3, 3)) not in ensembles


@pytest.mark.parametrize("ensembles", [[1, 2, 3]], indirect=True)
def test_get_item_when_contained_int(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should return the ensemble which exists and has the correct id
    """
    ids = [1, 2, 3]
    for id in ids:
        ensemble = ensembles[id]
        assert ensemble.id == id
        assert ensemble.dir == ensembles.context.join(ensembles.dir, str(id))
        assert ensemble.exists


@pytest.mark.parametrize("ensembles", [[(1, 1), (2, 2)]], indirect=True)
def test_get_item_when_contained_tuple(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should return the ensemble which exists and has the correct id for tuples
    """
    ids = [(1, 1), (2, 2)]
    for id in ids:
        ensemble = ensembles[id]
        assert ensemble.id == id
        assert ensemble.dir == ensembles.context.join(ensembles.dir, str(id))
        assert ensemble.exists


@pytest.mark.parametrize("ensembles", [[1, 2, 3]], indirect=True)
def test_get_item_when_not_contained(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should return the ensemble which exists and has the correct id
    """
    ids = [1, 2, 3]
    for id in ids:
        ensemble = ensembles[id]
        assert ensemble.id == id
        assert ensemble.dir == ensembles.context.join(ensembles.dir, str(id))
        assert ensemble.exists


@pytest.mark.parametrize("ensembles", [[(1, 1), (2, 2)]], indirect=True)
def test_get_item_when_contained_tuple(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should return a Run object but it should not exist
    """
    ensemble = ensembles[(3, 3)]
    assert not ensemble.exists


@pytest.mark.parametrize("ensembles", [[1, 2, 3]], indirect=True)
def test_iter_with_int(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should return an iterator over the ids which exist but as str

    Does not gaensembletee order based on listdir used to read ensembles
    """
    ids = set(["1", "2", "3"])

    for id in ensembles:
        assert id in ids

    assert ids == set(ensembles)
    assert ids == set(ensembles.__iter__())


@pytest.mark.parametrize("ensembles", [[(1, 1), (2, 2)]], indirect=True)
def test_iter_with_tuples(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should return an iterator over the ids which exist but as str

    Does not gaensembletee order based on listdir used to read ensembles
    """
    ids = set(["(1, 1)", "(2, 2)"])

    for id in set(ensembles):
        assert id in ids

    assert ids == set(ensembles.__iter__())
    assert ids == set(ensembles)


@pytest.mark.parametrize("ensembles", [[1, 2, 3]], indirect=True)
def test_works_as_map(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The Ensembles object to check

    Expects
    -------
    * Should work as a typical Mapping object does, iteration and unpacking
    """
    ids = set(["1", "2", "3"])

    for id in ids:
        assert id in ensembles

    for id in ensembles.keys():
        assert id in ids

    for ensemble in ensembles.values():
        assert ensemble.exists
        assert ensemble.id in ids

    for id, ensemble in ensembles.items():
        assert ensemble.id == id

    d = {**ensembles}
    assert dict(ensembles.items()) == d

    for id in d:
        assert id in ids

    for id, ensemble in d.items():
        assert ensemble == ensembles[id]


def test_save_targets(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The ensembles object to check

    Expects
    -------
    * Should save targets to file
    """
    targets = np.array([1,1,1])
    ensembles.save_targets(targets)
    assert ensembles.context.exists(ensembles.targets_path)


def test_load_targets(ensembles: Ensembles):
    """
    Parameters
    ----------
    ensembles: Ensembles
        The ensembles object to check

    Expects
    -------
    * Should be able to load targets from file
    * These loaded targets should be the same
    """
    targets = np.array([1,1,1])
    ensembles.save_targets(targets)

    loaded = ensembles.targets()
    assert all(targets == loaded)

