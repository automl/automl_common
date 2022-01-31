from typing import Callable

from pathlib import Path

from pytest_cases import fixture

from automl_common.backend import Backend


@fixture(scope="function")
def make_backend() -> Callable[[Path], Backend]:
    """
    Parameters
    ----------
    path: Path
        The path to create the backend at
    """

    def _make(path: Path) -> Backend:
        return Backend(name=path.name, path=path)

    return _make
