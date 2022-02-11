import numpy as np
from pytest_cases import parametrize

from automl_common.data.validate import jagged


@parametrize(
    "x, expected",
    [
        (np.array([1, 1, 1, 1]), False),
        (np.array([[1, 1], [1, 1]]), False),
        (np.array([[1, 2, 3], [1, 2]], dtype=object), True),
        (np.zeros((2, 2), dtype=object), False),
        (np.array([list([0, 0]), list([0, 0])], dtype=object), False),
    ],
)
def test_jagged(x: np.ndarray, expected: bool) -> None:
    assert jagged(x) is expected
