from typing import Any, Iterator, List, Optional, Union

import re
import shutil
from pathlib import Path
from tempfile import gettempdir

import pytest
from pytest import ExitCode, Item, Session

Config = Any
Parser = Any

# Load in other pytest modules, in this case fixtures
here = Path(__file__)
manual_tmp = Path(gettempdir()) / "test_automl_common"

pytest_plugins = []


def walk(
    path: Path,
    include: Optional[str] = None,
) -> Iterator[Path]:
    """Yeilds all files, iterating over directory"""
    for p in path.iterdir():
        if p.is_dir():
            if include is None or re.match(include, p.name):
                yield from walk(p, include)
        else:
            yield p.resolve()


def is_fixture(path: Path) -> bool:
    """Whether a path is a fixture"""
    return path.name.endswith("fixtures.py")


def is_factory(path: Path) -> bool:
    """Whether a path is a factory"""
    return path.name.endswith("factory.py")


def as_module(path: Path) -> str:
    """Convert a path to a module as seen from here"""
    root = here.parent.parent
    parts = path.relative_to(root).parts
    return ".".join(parts).replace(".py", "")


def fixture_modules() -> List[str]:
    """Get all fixture modules"""
    return [as_module(path) for path in walk(here.parent, include=r"test_(.*)") if is_fixture(path)]


def factory_modules() -> List[str]:
    """Get all fixture modules"""
    return [as_module(path) for path in walk(here.parent, include=r"test_(.*)") if is_factory(path)]


pytest_plugins += fixture_modules() + factory_modules()


def pytest_addoption(parser: Parser) -> None:
    """

    Parameters
    ----------
    parser : Parser
        The parser to add options to
    """
    parser.addoption(
        "--sklearn",
        action="store_true",
        default=False,
        help="Run sklearn compatibility tests",
    )


def pytest_sessionstart(session: Session) -> None:
    """Called after the ``Session`` object has been created and before performing collection
    and entering the run test loop.

    Parameters
    ----------
    session : Session
        The pytest session object
    """
    if manual_tmp.exists():
        shutil.rmtree(manual_tmp)

    manual_tmp.mkdir()


def pytest_sessionfinish(session: Session, exitstatus: Union[int, ExitCode]) -> None:
    """Called after whole test run finished, right before returning the exit status to the system.

    Parameters
    ----------
    session : Session
        The pytest session object.

    exitstatus: int | ExitCode
        The status which pytest will return to the system.
    """
    if manual_tmp.exists():
        shutil.rmtree(manual_tmp)


def pytest_runtest_setup(item: Item) -> None:
    """Run before each test"""
    todos = [mark for mark in item.iter_markers(name="todo")]
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")


def pytest_collection_modifyitems(
    session: Session,
    config: Config,
    items: List[Item],
) -> None:
    """Modifys the colelction of tests that are captured"""
    if config.getoption("--sklearn"):
        return

    skip_sklearn = pytest.mark.skip(reason="Need --sklearn option to run")
    for item in items:
        if "sklearn" in item.keywords:
            item.add_marker(skip_sklearn)
