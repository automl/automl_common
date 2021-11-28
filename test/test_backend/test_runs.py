import pytest

from pathlib import Path

from automl_common.backend.runs import Runs
from automl_common.backend.context import Context


def test_construction(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path that exists

    context: Context
        A context to interact with a filesystem

    Expects
    -------
    * Should construct without errors and set attributes
    """
    runs = Runs(dir=tmpdir, context=context)
    assert runs.dir == tmpdir
    assert runs.context == context


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_len_non_empty(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should have the same length as the number of runs it contains
    """
    assert len(runs) == 3


@pytest.mark.parametrize("runs", [[]], indirect=True)
def test_len_non_empty(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should have no length
    """
    assert len(runs) == 0


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_contains_with_run_contained_int(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should have the run with the id contained
    """
    assert all(i in runs for i in [1, 2, 3])
    assert all(str(i) in runs for i in [1, 2, 3])


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_contains_without_contained_run(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should give false for containing a run it does not contain
    """
    assert 4 not in runs
    assert str(4) not in runs


@pytest.mark.parametrize("runs", [[(1, 1), (2, 2)]], indirect=True)
def test_contains_with_run_contained_tuple(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should have the run with the id contained
    """
    assert (1, 1) in runs and (2, 2) in runs
    assert str((1, 1)) in runs and str((2, 2)) in runs


@pytest.mark.parametrize("runs", [[(1, 1), (2, 2)]], indirect=True)
def test_contains_without_contain_run_tuple(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should give false for run it does not contain
    """
    assert (3, 3) not in runs
    assert str((3, 3)) not in runs


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_get_item_when_contained_int(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should return the run which exists and has the correct id
    """
    ids = [1, 2, 3]
    for id in ids:
        run = runs[id]
        assert run.id == id
        assert run.dir == runs.context.join(runs.dir, str(id))
        assert run.exists


@pytest.mark.parametrize("runs", [[(1, 1), (2, 2)]], indirect=True)
def test_get_item_when_contained_tuple(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should return the run which exists and has the correct id for tuples
    """
    ids = [(1, 1), (2, 2)]
    for id in ids:
        run = runs[id]
        assert run.id == id
        assert run.dir == runs.context.join(runs.dir, str(id))
        assert run.exists


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_get_item_when_not_contained(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should return the run which exists and has the correct id
    """
    ids = [1, 2, 3]
    for id in ids:
        run = runs[id]
        assert run.id == id
        assert run.dir == runs.context.join(runs.dir, str(id))
        assert run.exists


@pytest.mark.parametrize("runs", [[(1, 1), (2, 2)]], indirect=True)
def test_get_item_when_contained_tuple(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should return a Run object but it should not exist
    """
    run = runs[(3, 3)]
    assert not run.exists


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_iter_with_int(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should return an iterator over the ids which exist but as str

    Does not garuntee order based on listdir used to read runs
    """
    ids = set(["1", "2", "3"])

    for id in runs:
        assert id in ids

    assert ids == set(runs)
    assert ids == set(runs.__iter__())


@pytest.mark.parametrize("runs", [[(1, 1), (2, 2)]], indirect=True)
def test_iter_with_tuples(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should return an iterator over the ids which exist but as str

    Does not garuntee order based on listdir used to read runs
    """
    ids = set(["(1, 1)", "(2, 2)"])

    for id in set(runs):
        assert id in ids

    assert ids == set(runs.__iter__())
    assert ids == set(runs)


@pytest.mark.parametrize("runs", [[1, 2, 3]], indirect=True)
def test_works_as_map(runs: Runs):
    """
    Parameters
    ----------
    runs: Runs
        The Runs object to check

    Expects
    -------
    * Should work as a typical Mapping object does, iteration and unpacking
    """
    ids = set(["1", "2", "3"])

    for id in ids:
        assert id in runs

    for id in runs.keys():
        assert id in ids

    for run in runs.values():
        assert run.exists
        assert run.id in ids

    for id, run in runs.items():
        assert run.id == id

    d = {**runs}
    assert dict(runs.items()) == d

    for id in d:
        assert id in ids

    for id, run in d.items():
        assert run == runs[id]
