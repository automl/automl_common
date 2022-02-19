from automl_common.backend.accessors.accessor import Accessor

import pytest
import test.test_backend.test_accessors.cases as cases
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases


@parametrize_with_cases("accessor", cases=cases)
def test_path(accessor: Accessor) -> None:
    """
    accessor: Accessor
        The accessor to test it's path

    Expects
    -------
    * Should return a path to an obj
    """
    assert accessor.path is not None


@parametrize_with_cases("accessor", cases=cases, filter=ft.has_tag("populated"))
def test_exist_populated(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with a populated obj

    Expects
    -------
    * Should have it's exist give that it exists
    """
    assert accessor.exists()


@parametrize_with_cases("accessor", cases=cases, filter=~ft.has_tag("populated"))
def test_exist_not_populated(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with no obj

    Expects
    -------
    * Should have that it doesn't exist
    """
    assert not accessor.exists()


@parametrize_with_cases("accessor", cases=cases, filter=ft.has_tag("populated"))
def test_load_populated(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with an obj stored

    Expects
    -------
    * Should be able to load without issue
    """
    assert accessor.load() is not None


@parametrize_with_cases("accessor", cases=cases, filter=~ft.has_tag("populated"))
def test_load_unpopulated(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with an obj stored

    Expects
    -------
    * Should give a file not found error
    """
    with pytest.raises(FileNotFoundError):
        accessor.load()


@parametrize_with_cases("accessor", cases=cases)
def test_save(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with or without an obj stored

    Expects
    -------
    * The saved obj should be the same one loaded back
    """
    # Works while we only use pickle load and save
    obj = {"hello": "world"}
    accessor.save(obj)
    assert accessor.load() == obj


@parametrize_with_cases("accessor", cases=cases, filter=ft.has_tag("populated"))
def test_delete(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with an obj stored

    Expects
    -------
    * Delete without folder=False should just delete the model
    """
    contents_before = set(accessor.dir.iterdir())

    accessor.delete(folder=False)

    contents_after = set(accessor.dir.iterdir())

    assert contents_after == contents_before - {accessor.path}


@parametrize_with_cases("accessor", cases=cases, filter=~ft.has_tag("populated"))
def test_delete_unpopulated(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor with no obj stored

    Expects
    -------
    * Delete when nothing is stored should give a FileNotFoundError
    """
    with pytest.raises(FileNotFoundError):
        accessor.delete(folder=False)


@parametrize_with_cases("accessor", cases=cases)
def test_delete_folder(accessor: Accessor) -> None:
    """
    accessor: Accessor
        An accessor obj

    Expects
    -------
    * Delete with folder=True should delete the entire folder
    """
    accessor.delete(folder=True)
    assert not accessor.dir.exists()


@parametrize_with_cases("accessor", cases=cases)
def test_str_has_dir(accessor: Accessor) -> None:
    """
    Parameters
    ----------
    accessor : Accessor
        The accessor to test

    Expects
    -------
    * The accessor should contain the directory in it's str output
    """
    assert str(accessor.dir) in str(accessor)
