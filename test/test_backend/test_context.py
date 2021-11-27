"""Tests the Contexts

If adding more contexts, add a fixture such as `local_context` and then
add it's string to `_contexts_to_tests`

*   The `tmpfile` and `tmpdir` fixture parameter will need to be updated
    once considering non-local context
"""
from typing import Union, List

import os

from pathlib import Path

import pytest
from pytest_lazyfixture import lazy_fixture  # Allows fixture in parametrization

from automl_common.backend.context import Context, LocalContext


def test_local_context_construction():
    """Should construct with no issues"""
    LocalContext()


@pytest.fixture(scope="function")
def local_context() -> LocalContext:
    return LocalContext()


_contexts = ["local_context"]


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
@pytest.mark.parametrize("mode, content", [("", "hello"), ("b", b"hello")])
def test_context_open(tmpfile: Path, context: Context, mode: str, content: Union[str, bytes]):
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a tmpfile that can be used

    context: Context
        A context object to test

    mode: "" | "b"
        The mode to read and write in

    content: str | bytes
        The context to write to file

    Expects
    -------
    * Should be able to write a file in both modes
    * Should be able to read a file in both modes
    """
    with context.open(tmpfile, "w" + mode) as f:
        f.write(content)

    assert tmpfile.exists()

    with context.open(tmpfile, "r" + mode) as f:
        assert f.read() == content


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_mkdir(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path to a tmpdir

    context: Context
        A context object to test

    Expects
    -------
    * Should create a directory if it doesn't exist
    """
    folder = tmpdir / "folder"
    context.mkdir(folder)

    assert folder.exists()


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_mkdir_fails_if_exists(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path to a tmpdir

    local_context: LocalContext
        A local_context object to check

    Expects
    -------
    * Should fail creating a dir if it already exists
    """
    folder = tmpdir / "folder"
    context.mkdir(folder)

    with pytest.raises(FileExistsError):
        context.mkdir(folder)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_makedirs_makes_dirs(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path to a tmpdir

    context: Context
        A local_context object to check

    Expects
    -------
    * Should create all dirs if the don't exist
    """
    path = tmpdir / "folder1" / "folder2"
    context.makedirs(path)

    assert context.exists(path)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_makedirs_fails_if_exists(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path to a tmpdir

    context: Context
        A context object to check

    Expects
    -------
    * Should not create dir if it exists
    """
    path = tmpdir / "folder1" / "folder2"
    context.makedirs(path, exist_ok=False)

    with pytest.raises(FileExistsError):
        context.makedirs(path)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_makedirs_if_partial_path_exists(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A path to a tmpdir

    context: Context
        A context object to check

    Expects
    -------
    * Should create full path even if it already partially exists
    """
    partial_path = tmpdir / "folder1"
    path = partial_path / "folder2"

    context.makedirs(partial_path)
    context.makedirs(path, exist_ok=True)

    assert path.exists()


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_exists(tmpfile: Path, context: Context):
    """
    Parameters
    ----------
    tmpfile: Path
        A path to a tmpdir

    context: LocalContext
        A context object to check

    Expects
    -------
    * Should give false for a file which does not exist
    * Should give true for a file that exists
    """
    assert not tmpfile.exists() and not context.exists(tmpfile)

    with context.open(tmpfile, "w") as f:
        f.write("Hello")

    assert tmpfile.exists() and context.exists(tmpfile)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rm(tmpfile: Path, context: Context):
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: Context
        The context object to check

    Expects
    -------
    * Should remove an existing file
    """
    with context.open(tmpfile, "w") as f:
        f.write("hello")

    assert tmpfile.exists()

    context.rm(tmpfile)

    assert not tmpfile.exists()


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rm_fails_if_not_exist(tmpfile: Path, context: Context):
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: Context
        The context object to check

    Expects
    -------
    * Should raise an error if trying to remove a file that doesn't exist
    """
    assert not tmpfile.exists()

    with pytest.raises(FileNotFoundError):
        context.rm(tmpfile)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rm_fails_with_dir(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to a temporary directory

    context: Context
        The context object to check

    Expects
    -------
    * Should raise an error if trying to remove a directory
    """
    assert tmpdir.exists()

    with pytest.raises(IsADirectoryError):
        context.rm(tmpdir)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rmdir(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to a temporary directory

    context: Context
        The context object to check

    Expects
    -------
    * Should delete a directory that exists
    """
    folder = tmpdir / "folder"

    context.mkdir(folder)
    assert folder.exists()

    context.rmdir(folder)

    assert not folder.exists()


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rmdir_fail_if_not_exist(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to a temporary file

    context: Context
        The context object to check

    Expects
    -------
    * Should fail if given a path that doesn't exist
    """
    folder = tmpdir / "folder"
    assert not folder.exists()

    with pytest.raises(FileNotFoundError):
        context.rmdir(folder)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rmdir_not_remove_file(tmpfile: Path, context: Context):
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: Context
        The context object to check

    Expects
    -------
    * Should not delete a file that exists
    """
    with context.open(tmpfile, "w") as f:
        f.write("hello")

    assert tmpfile.exists()

    with pytest.raises(NotADirectoryError):
        context.rmdir(tmpfile)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_context_rmdir_not_remove_file(tmpfile: Path, context: Context):
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: Context
        The context object to check

    Expects
    -------
    * Should not delete a file that exists
    """
    with context.open(tmpfile, "w") as f:
        f.write("hello")

    assert tmpfile.exists()

    with pytest.raises(NotADirectoryError):
        context.rmdir(tmpfile)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_tmpdir_creates_tmpdir(context: Context):
    """
    Parameters
    ----------
    context: Context
        The context object to check

    Expects
    -------
    * Should create a tmp directory in the context and delete it after
    """
    tmp_obj = None
    with context.tmpdir() as tmp:
        tmp_obj = tmp  # Store a reference to the path
        assert context.exists(tmp)

    assert not context.exists(tmp)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
@pytest.mark.parametrize("prefix", ["pre", "__", "test."])
def test_tmpdir_creates_tmpdir_with_prefix(context: Context, prefix: str):
    """
    Parameters
    ----------
    context: Context
        The context object to check

    prefix: str
        A prefix to add

    Expects
    -------
    * Should create a tmp directory and it should begin with the prefix
    """
    with context.tmpdir(prefix=prefix) as tmp:
        assert context.exists(tmp)
        assert tmp.name.startswith(prefix)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
@pytest.mark.parametrize("prefix", ["pre", "__", "test."])
def test_tmpdir_creates_tmpdir_with_prefix(context: Context, prefix: str):
    """
    Parameters
    ----------
    context: Context
        The context object to check

    prefix: str
        A prefix to add

    Expects
    -------
    * Should create a tmp directory and it should begin with the prefix
    """
    with context.tmpdir(prefix=prefix) as tmp:
        assert context.exists(tmp)

        assert tmp.name.startswith(prefix), tmp.name


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_tmpdir_retains(context: Context):
    """
    Parameters
    ----------
    context: Context
        The context object to check

    Expects
    -------
    * Should not delete the tmpdir after it context manager ends
    """
    tmp_object = None
    with context.tmpdir(retain=True) as tmp:
        tmp_object = tmp  # Hold reference
        assert context.exists(tmp)

    assert context.exists(tmp_object)


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
@pytest.mark.parametrize(
    "segments",
    [
        ("one",),
        (
            "one",
            "two",
        ),
        (
            "one",
            "two",
            "three",
        ),
    ],
)
def test_join(context: Context, segments: List[str]):
    """
    Parameters
    ----------
    context: Context
        The context to test

    segments: Tuple[str]
        A collection of strings

    Expects
    -------
    * Join should work on arbitrary length of parameters and be in the right order

    Can't make any assumptions on what joins two segments of a path
    """
    print(*segments)
    path = context.join(*segments)
    assert isinstance(path, str)

    for segment in segments:
        assert segment in path

    if len(segments) > 1:
        for i in range(len(segments) - 1):
            assert path.find(segment[i]) < path.find(segment[i + 1])


@pytest.mark.parametrize("context", lazy_fixture(_contexts))
def test_listdir_gives_lists_files_and_folders(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        A tmpdir to use

    context: Context
        The context object to test

    Expects
    -------
    * All files and folders created will show up in listdir
    """
    files = ["one", "two", "three"]
    folders = ["a", "b", "c"]

    for file in files:
        path = context.join(tmpdir, file)
        with context.open(path, "w") as f:
            f.write("hello")

    for folder in folders:
        path = context.join(tmpdir, folder)
        context.mkdir(path)

    listed = context.listdir(tmpdir)
    assert all(file in listed for file in files)
    assert all(folder in listed for folder in folders)
