from typing import Any

from pathlib import Path

import pytest

from automl_common.backend import Context, DataManager


def test_construction(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to existing tmpdir

    context: Context
        Context to access file system

    Expects
    -------
    * Should set it's init properties properly
    """
    dm = DataManager(dir=tmpdir, context=context)
    assert dm.dir == tmpdir
    assert dm.context == context


def test_data_path_property(datamanager: DataManager):
    """
    Parameters
    ----------
    datamanager: DataManager
        The datamanager to test

    Expects
    -------
    * Should be based off the root directory it was created with
    """
    expected = datamanager.context.join(datamanager.dir, "datamanager.pkl")
    assert datamanager.data_path == expected


@pytest.mark.parametrize("data", [(1, 2, 3), {"hello": "world"}, 10])
def test_saves(datamanager: DataManager, data: Any):
    """
    Parameters
    ----------
    datamanager: DataManager
        The datamanager to test

    data: Any
        Data to be pickled. Not sure what type picklable data is or it's requirements

    Expects
    -------
    * Should successfully save picklable objects
    """
    datamanager.save(data)
    assert datamanager.context.exists(datamanager.data_path)


@pytest.mark.parametrize("data", [(1, 2, 3), {"hello": "world"}, 10])
def test_datamanager_loads(datamanager: DataManager, data: Any):
    """
    Parameters
    ----------
    datamanager: DataManager
        The datamanager to test

    data: Any
        Data to be pickled. Not sure what type picklable data is or it's requirements

    Expects
    -------
    * Should successfully reload the data it pickled
    """
    datamanager.save(data)
    assert datamanager.context.exists(datamanager.data_path)

    loaded = datamanager.load()
    assert loaded == data
