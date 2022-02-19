from typing import Iterable, List, Set

from pytest_cases import parametrize

from automl_common.util.functional import dictmerge, intersection, union


@parametrize(
    "xs, expected",
    [
        [([1, 2, 3], [2, 3], []), set()],
        [([1, 2, 3], [2, 3], [2]), {2}],
        [([1, 2, 3], [3], [2, 3]), {3}],
        [{}, set()],
    ],
)
def test_intersection(xs: Iterable[List[int]], expected: Set[int]) -> None:
    assert intersection(xs) == expected


@parametrize(
    "xs, expected",
    [
        [([1, 2, 3], [2, 3], []), {1, 2, 3}],
        [([1], [2], [3]), {1, 2, 3}],
        [([1], [], [3]), {1, 3}],
        [{}, set()],
    ],
)
def test_union(xs: Iterable[List[int]], expected: Set[int]) -> None:
    assert union(xs) == expected


def test_dictmerge() -> None:
    a = {"a": "hello", "b": "world", "c": "mars"}
    b = {"x": "hello", "b": "world", "c": "venus"}
    expected = {
        "a": "hello",
        "x": "hello",
        "b": ["world", "world"],
        "c": ["mars", "venus"]
    }

    assert dictmerge([a, b, {}]) == expected
