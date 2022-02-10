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
