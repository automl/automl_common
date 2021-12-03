import logging

import pytest

from pytest_lazyfixture import lazy_fixture

from automl_common.backend import Backend, LocalContext, Context


def test_default_construction_set_properties():
    """
    Expects
    -------
    * Should set its root
    * Should create the root
    * The framework default param "framework" should be in the root
    * Should set a default framework of "framework"
    * Should have a context of LocalContext
    * Should set retain to False
    * Should create the private objects to access further backend parts
    * Should have the logger be None by default
    * Should create the folders for the other backend objects
    * Should have root in the paths to the other backend objects
    """
    backend = Backend()

    assert backend._root is not None
    assert backend.exists(backend._root)

    assert backend._framework == "framework"
    assert "framework" in backend._root

    assert isinstance(backend._context, LocalContext)

    assert backend._retain == False

    assert backend._logger is None

    backend_objs = [
        backend.optimizer,
        backend.ensembles,
        backend.runs,
        backend.datamanager,
    ]
    for obj in backend_objs:
        assert obj is not None


    paths =[
        backend.framework_dir,
        backend.optimizer_dir,
        backend.ensembles_dir,
        backend.runs_dir,
        backend.data_dir,
    ]
    for path in paths:
        assert backend.exists(path)

    for path in paths:
        assert backend._root in path


def test_construction_fails_when_root_exists(tmpdir: str):
    """
    Parameters
    ----------
    tmpdir: str
        Path to already existing directory

    Expects
    -------
    * Should raise a RuntimeError that the path already exists
    """
    with pytest.raises(RuntimeError):
        backend = Backend(root=tmpdir)


@pytest.mark.parametrize("framework", ["autosklearn", "autopytorch"])
def test_construction_with_framework_is_in_root_path_when_no_dir(framework: str):
    """
    Parameters
    ----------
    framework: str
        The framework name

    Expects
    -------
    * Should put the `framework` in the root path when no `dir` is specified
    """
    backend = Backend(framework=framework)
    assert backend._framework == framework
    assert framework in backend._root


@pytest.mark.parametrize("root", [lazy_fixture("tmpfile")])
def test_construction_with_root(root: str):
    """
    Parameters
    ----------
    root: str
        The temporary dir name

    Expects
    -------
    * Should set the root correctly and create it
    """
    backend = Backend(root=root)
    assert backend._root == root
    assert backend.exists(root)


def test_construction_with_context(context: Context):
    """
    Parameters
    ----------
    context: Context
        The context to use

    Expects
    -------
    * Should set the context property correctly
    """
    backend = Backend(context=context)
    assert backend._context == context


@pytest.mark.parametrize("retain", [True, False])
def test_del_with_retain(retain: bool):
    """
    Parameters
    ----------
    retain: bool
        Whether the root should be delete or not

    Expects
    -------
    * Should delete the root folder if True, else keep it
    """
    backend = Backend(retain=retain)

    root = backend._root
    context = backend._context  # Used to check existence

    assert backend.exists(root)

    del backend
    assert context.exists(root) == retain


def test_logger_property_gives_default_logger(backend: Backend):
    """
    Parameters
    ----------
    backend: Backend
        The backend to test

    Expects
    -------
    * There should be no private logger set before property being called
    * The default logger should be on localhost
    * The port should be DEFAULT_TCP_LOGGING_PORT
    * The name should be `backend`
    """
    assert backend._logger is None

    logger = backend.logger
    assert logger is not None and backend._logger == logger

    assert logger.name == "automl_common.backend.backend"
    assert logger.host == "localhost"
    assert logger.port == logging.handlers.DEFAULT_TCP_LOGGING_PORT



@pytest.mark.parametrize("port", [1338, 1339, 1440])
def test_setup_logger_sets_logger(backend: Backend, port: int):
    """
    Note:
        Not sure if this might be an issue if the port is already taken.
        Could check if the port is taken before constructing

    Parameters
    ----------
    backend: Backend
        The backend to test

    port: Port
        The port for the logger to operate on

    Expects
    -------
    * There should be no private logger set before property being called
    * The logger created should have a port the same as the parameter
    """
    assert backend._logger is None

    backend.setup_logger(port)
    logger = backend.logger

    assert logger is not None and backend._logger == logger

    assert logger.port == port


def test_mark_start(backend: Backend):
    """
    Parameters
    ----------
    backend: Backend
        The backend object to test

    Expects
    -------
    * There should be no start time before marking
    * Should write the time to the the path backend.start_time_path
    """
    backend = Backend()
    assert not backend.exists(backend.start_time_path)

    backend.mark_start()
    assert backend.exists(backend.start_time_path)


def test_mark_start_fails_when_already_marked(backend: Backend):
    """
    Parameters
    ----------
    backend: Backend
        The backend object to test

    Expects
    -------
    * Should raise a RuntimeError when the start has already been marked
    """
    backend = Backend()
    backend.mark_start()

    with pytest.raises(RuntimeError):
        backend.mark_start()


def test_mark_end(backend: Backend):
    """
    Parameters
    ----------
    backend: Backend
        The backend object to test

    Expects
    -------
    * There should be no end time before marking
    * Should write the time to the the path backend.end_time_path
    """
    backend = Backend()
    assert not backend.exists(backend.end_time_path)

    backend.mark_end()
    assert backend.exists(backend.end_time_path)


def test_mark_end_fails_when_already_marked(backend: Backend):
    """
    Parameters
    ----------
    backend: Backend
        The backend object to test

    Expects
    -------
    * Should raise a RuntimeError when the end has already been marked
    """
    backend = Backend()
    backend.mark_end()

    with pytest.raises(RuntimeError):
        backend.mark_end()


#################################################################
# The tests for context functions are covered by context. This is
# because backend follows the same API as context.
#################################################################
