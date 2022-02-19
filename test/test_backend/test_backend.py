from typing import Optional

from pathlib import Path

from automl_common.backend import Backend

from pytest_cases import fixture_ref, parametrize


def test_makes_non_existent_dir(path: Path) -> None:
    """
    Parameters
    ----------
    path: Path
        The path to base from

    Expects
    -------
    * Should create the path if it does not exist
    """
    name = "test_automl_common"
    path = path / "dir"
    backend = Backend(name=name, path=path)

    assert backend.path.exists()


def test_makes_non_existent_nested_dir(path: Path) -> None:
    """
    Parameters
    ----------
    path: Path
        The path to base from

    Expects
    -------
    * Should create the path if it does not exist, even when nestesd
    """
    name = "test_automl_common"
    path = path / "dir" / "nested"
    backend = Backend(name=name, path=path)

    assert backend.path.exists()


def test_default_gives_tmpfolder_with_name_prefix() -> None:
    """
    Expects
    -------
    * Backend should default to give a tmpdir with the name as the prefix
    """
    name = "test_automl_common"
    backend = Backend(name=name)

    assert backend.path.name.startswith(name)


def test_str_path_gets_converted_to_local_pathlib_path(path: Path) -> None:
    """
    Parameters
    ----------
    path: Path
        The path which will be given as a string
    """
    name = "test_automl_common"
    strpath = str(path)
    backend = Backend(name=name, path=strpath)

    assert isinstance(backend.path, Path)


@parametrize("target_path, should_retain", [(fixture_ref("path"), True), (None, False)])
def test_default_retain_behaviour(
    target_path: Optional[Path],
    should_retain: bool,
) -> None:
    """
    Parameters
    ----------
    target_path: Optional[Path]
        The path to use, optionally

    should_retain: bool
        Whether the backend should retain given the target path
    """
    name = "test_automl_common"
    backend = Backend(name=name, path=target_path)

    assert backend.retain == should_retain

    path = backend.path
    del backend

    assert path.exists() == should_retain


@parametrize("target_path", [fixture_ref("path"), None])
@parametrize("retain", [True, False])
def test_retain_parameter(target_path: Path, retain: bool) -> None:
    """
    Parameters
    ----------
    target_path: Optional[Path]
        The path to use, optionally

    retain: bool
        Should the backend be retained
    """
    name = "test_automl_common"
    backend = Backend(name=name, path=target_path, retain=retain)

    assert backend.retain == retain

    path = backend.path
    del backend

    assert path.exists() == retain
