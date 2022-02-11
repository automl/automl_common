from typing import Iterator, List, Optional

import re
from pathlib import Path

# Load in other pytest modules, in this case fixtures
here = Path(__file__)

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
