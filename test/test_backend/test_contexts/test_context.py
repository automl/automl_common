"""Tests the Contexts

*   The `tmpfile` and `tmp_path` fixture parameter will need to be updated
    once considering non-local context
"""
from typing import Union

from pathlib import Path

import pytest
from pytest_cases import parametrize

from automl_common.backend.contexts import OSContext


def test_construction() -> None:
    """

    Expects
    -------
    * Constructs without issue
    """
    OSContext()


@parametrize("mode, content", [("", "hello"), ("b", b"hello")])
def test_open(
    tmpfile: Path,
    context: OSContext,
    mode: str,
    content: Union[str, bytes],
) -> None:
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a tmpfile that can be used

    context: OSContext
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


def test_context_mkdir(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        A path to a tmp_path

    context: OSContext
        A context object to test

    Expects
    -------
    * Should create a directory if it doesn't exist
    """
    folder = tmp_path / "folder"
    context.mkdir(folder)

    assert folder.exists()


def test_context_mkdir_fails_if_exists(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        A path to a tmp_path

    context: OSContext
        A local_context object to check

    Expects
    -------
    * Should fail creating a dir if it already exists
    """
    folder = tmp_path / "folder"
    context.mkdir(folder)

    with pytest.raises(FileExistsError):
        context.mkdir(folder)


def test_context_makedirs_makes_dirs(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        A path to a tmp_path

    context: OSContext
        A local_context object to check

    Expects
    -------
    * Should create all dirs if the don't exist
    """
    path = tmp_path / "folder1" / "folder2"
    context.makedirs(path)

    assert path.exists()


def test_context_makedirs_fails_if_exists(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        A path to a tmp_path

    context: OSContext
        A context object to check

    Expects
    -------
    * Should not create dir if it exists
    """
    path = tmp_path / "folder1" / "folder2"
    context.makedirs(path, exist_ok=False)

    with pytest.raises(FileExistsError):
        context.makedirs(path)


def test_context_makedirs_if_partial_path_exists(
    tmp_path: Path,
    context: OSContext,
) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        A path to a tmp_path

    context: OSContext
        A context object to check

    Expects
    -------
    * Should create full path even if it already partially exists
    """
    partial_path = tmp_path / "folder1"
    path = partial_path / "folder2"

    context.makedirs(partial_path)
    context.makedirs(path, exist_ok=True)

    assert path.exists()


def test_context_exists(tmpfile: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmpfile: Path
        A path to a tmp_path

    context: OSContext
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


def test_context_rm(tmpfile: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: OSContext
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


def test_context_rm_fails_if_not_exist(tmpfile: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: OSContext
        The context object to check

    Expects
    -------
    * Should raise an error if trying to remove a file that doesn't exist
    """
    assert not tmpfile.exists()

    with pytest.raises(FileNotFoundError):
        context.rm(tmpfile)


def test_context_rm_fails_with_dir(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        Path to a temporary directory

    context: OSContext
        The context object to check

    Expects
    -------
    * Should raise an error if trying to remove a directory
    """
    assert tmp_path.exists()

    with pytest.raises(IsADirectoryError):
        context.rm(tmp_path)


def test_context_rmdir(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        Path to a temporary directory

    context: OSContext
        The context object to check

    Expects
    -------
    * Should delete a directory that exists
    """
    folder = tmp_path / "folder"

    context.mkdir(folder)
    assert folder.exists()

    context.rmdir(folder)

    assert not folder.exists()


def test_context_rmdir_fail_if_not_exist(tmp_path: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        Path to a temporary file

    context: OSContext
        The context object to check

    Expects
    -------
    * Should fail if given a path that doesn't exist
    """
    folder = tmp_path / "folder"
    assert not folder.exists()

    with pytest.raises(FileNotFoundError):
        context.rmdir(folder)


def test_context_rmdir_not_remove_file(tmpfile: Path, context: OSContext) -> None:
    """
    Parameters
    ----------
    tmpfile: Path
        Path to a temporary file

    context: OSContext
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


def test_tmpdir_creates_tmp_path(context: OSContext) -> None:
    """
    Parameters
    ----------
    context: OSContext
        The context object to check

    Expects
    -------
    * Should create a tmp directory in the context and delete it after
    """
    tmp_obj = None
    with context.tmpdir() as tmp:
        tmp_obj = tmp  # Store a reference to the path
        assert tmp.exists()

    assert not tmp_obj.exists()


@pytest.mark.parametrize("prefix", ["pre", "__", "test."])
def test_tmp_path_creates_tmp_path_with_prefix(context: OSContext, prefix: str) -> None:
    """
    Parameters
    ----------
    context: OSContext
        The context object to check

    prefix: str
        A prefix to add

    Expects
    -------
    * Should create a tmp directory and it should have the prefix in it
    """
    with context.tmpdir(prefix=prefix) as tmp:
        assert tmp.exists()
        assert tmp.name.startswith(prefix)


def test_tmp_path_retains(context: OSContext) -> None:
    """
    Parameters
    ----------
    context: OSContext
        The context object to check

    Expects
    -------
    * Should not delete the tmp_path after it context manager ends
    """
    tmp_object = None
    with context.tmpdir(retain=True) as tmp:
        tmp_object = tmp  # Hold reference
        assert tmp.exists()

    assert context.exists(tmp_object)


def test_listdir_gives_lists_files_and_folders(
    tmp_path: Path,
    context: OSContext,
) -> None:
    """
    Parameters
    ----------
    tmp_path: Path
        A tmp_path to use

    context: OSContext
        The context object to test

    Expects
    -------
    * All files and folders created will show up in listdir
    """
    files = ["one", "two", "three"]
    folders = ["a", "b", "c"]

    for file in files:
        path = tmp_path / file
        with context.open(path, "w") as f:
            f.write("hello")

    for folder in folders:
        path = tmp_path / folder
        context.mkdir(path)

    listed = context.listdir(tmp_path)
    assert all(file in listed for file in files)
    assert all(folder in listed for folder in folders)


@parametrize("strpath", ["/home/user/somewhere", "nothere/there/out.txt"])
def test_as_path(strpath: str, context: OSContext) -> None:
    """
    Parameters
    ----------
    strpath: str
        The path to convert

    context: OSContext
        The context to use

    Expects
    -------
    * Should return a path object given a str path
    """
    path = context.as_path(strpath)
    assert isinstance(path, Path)