from typing import Iterable, List, Tuple

import numpy as np
import pytest
from pytest_cases import case, parametrize_with_cases, parametrize

from automl_common.data.math import majority_vote, weighted_sum

from test.data import DEFAULT_SEED


@case(tags=["weighted_sum"])
def case_weighted_sum_one() -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    arrays = [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
    weights = np.array([1, 10, 100])
    expected = np.array([111, 111, 111])
    return arrays, weights, expected


@parametrize_with_cases("arrays, weights, expected", cases=".", has_tag="weighted_sum")
def test_weighted_sum_arrays(
    arrays: Iterable[np.ndarray],
    weights: np.ndarray,
    expected: np.ndarray,
) -> None:
    """

    Parameters
    ----------
    arrays : Iterable[np.ndarray], (n, m)
        An iterable of arrays to sum

    weights : np.ndarray (m, )
        The weights to multiply each array by

    expected : np.ndarray (m, )
        The expected result

    Expects
    -------
    * Should produce the expected result
    """
    result = weighted_sum(arrays, weights)
    np.testing.assert_equal(result, expected)


@case(tags=["majority_vote"])
def case_majority_vote_one() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.array(
        [
            ["a", "b", "a"],
            ["b", "b", "c"],
            ["c", "c", "b"],
            ["a", "b", "b"],
        ]
    ).T
    weights = np.array([0.1, 0.2, 0.8])
    expected = np.array(["a", "c", "b", "b"])
    return arrays, weights, expected


@case(tags=["majority_vote"])
def case_majority_vote_two() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.array(
        [
            ["a", "b", "a"],
            ["b", "b", "c"],
            ["c", "c", "b"],
            ["a", "b", "b"],
        ]
    ).T
    weights = np.array([2, 2, 3])
    expected = np.array(["a", "b", "c", "b"])
    return arrays, weights, expected


@parametrize_with_cases("arrays, weights, expected", cases=".", has_tag="majority_vote")
def test_majority_vote(arrays: np.ndarray, weights: np.ndarray, expected: np.ndarray) -> None:
    """
    Parameters
    ----------
    arrays : np.ndarray, (n, m)
        The arrays of votes

    weights : np.ndarray (m,)
        The weights to apply to each voter

    expected : np.ndarray (m,)
        The expected result

    Expects
    -------
    * Majority vote will produce the expected result
    """
    result = majority_vote(arrays, weights=weights)
    np.testing.assert_equal(result, expected)


@parametrize("n, m", [(10, 3), (5, 5)])
def test_majority_vote_with_bad_shapes(m: int, n: int) -> None:
    shape = (n, m)
    weight_shape = (m + 1,)
    rs = np.random.RandomState(DEFAULT_SEED)

    arrays = rs.random(shape)
    weights = rs.random(weight_shape)

    with pytest.raises(ValueError):
        majority_vote(arrays, weights)

    return  # pragma: no cover
