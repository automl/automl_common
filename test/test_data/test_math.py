from typing import Iterable, Iterator, List, Tuple

import numpy as np
from pytest_cases import case, parametrize_with_cases

from automl_common.data.math import weighted_sum


@case
def case_lists() -> Tuple[List[int], List[np.ndarray], np.ndarray]:
    weights = [1, 10, 100]
    arrays = [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
    expected = np.array([111, 111, 111])
    return weights, arrays, expected


@case
def case_iters() -> Tuple[Iterator[float], Iterator[np.ndarray], np.ndarray]:
    weights = iter([-1.0, 0.5, 0.5])
    arrays = iter([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])])
    expected = np.array([0, 0, 0])
    return weights, arrays, expected


@parametrize_with_cases("weights, arrays, expected", cases=".")
def test_weighted_sum_arrays(
    weights: Iterable[float],
    arrays: Iterable[np.ndarray],
    expected: np.ndarray,
) -> None:
    result = weighted_sum(weights, arrays)
    np.testing.assert_array_equal(result, expected)
