from typing import List, Optional

from pathlib import Path

from automl_common.backend.util.path import rmtree

from pytest_cases import parametrize


@parametrize("contents", [None, ["file.txt", "other"]])
def test_rmtree_flat(path: Path, contents: Optional[List[str]]) -> None:
    """Should delete the folder

    Parameters
    ----------
    path: Path
        The path to base from

    contents: Optional[List[str]]
        Contents of the folder

    Expects
    -------
    * The path should not exist after rmtree
    """
    test_path = path / "populated"
    test_path.mkdir()

    if contents is not None:
        for filename in contents:
            path.joinpath(filename).touch()

    rmtree(test_path)
    assert not test_path.exists()


def test_rmtree_nested(path: Path) -> None:
    """Should delete the folder

    Parameters
    ----------
    path: Path
        The path to base from

    Expects
    -------
    * The path should not exist after rmtree
    """
    test_path = path / "populated"
    test_path.mkdir()

    nested_dir = test_path / "nested"
    nested_dir.mkdir()

    further_nested = nested_dir / "further_nested"
    further_nested.mkdir()

    rmtree(test_path)
    assert not test_path.exists()
