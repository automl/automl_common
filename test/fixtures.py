from typing import Callable

from pathlib import Path

from pytest import FixtureRequest
from pytest_cases import fixture, fixture_ref, parametrize


@fixture(scope="function")
@parametrize("impl", [fixture_ref("tmp_path")])
def path(request: FixtureRequest, impl: Path) -> Path:
    """Forwards different kinds of paths we might want to use

    In the future we may want an AWSPath: Path to run tests with

    The should delete themselves after use
    """
    return impl


@fixture(scope="function")
@parametrize("impl", [fixture_ref("tmp_path")])
def make_path(impl: Path) -> Callable[[str], Path]:
    """Returns a factory for paths

    def test(make_path: Callable[[str], Path]):
        path1 = make_path("hello")
        path2 = make_path("world")

    Will use any kinds of paths included in its @parametrize

    The should delete themselves after use
    """

    def make(name: str) -> Path:
        return impl / name

    return make
