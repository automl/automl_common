from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.ensemble import WeightedEnsemble

import test.test_ensemble.cases as cases


def test_empty_ids(path: Path) -> None:
    """
    Parameters
    ----------
    path : Path
        A dummy path to use

    Expects
    -------
    * Should raise a value error if the ids is empty
    """
    with pytest.raises(ValueError):
        WeightedEnsemble(path, weighted_ids={})


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("valid"),
)
def test_predict(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble: WeightedEnsemble
        A weighted ensemble with weights

    Expects
    -------
    * Weighted_sum function should recieve the weights in the correct order with the
      predictions.
    """
    x = np.asarray([1, 10, 100])

    ids, weights = zip(*ensemble.weights.items())
    expected_predictions = iter(ensemble[id].predict(x) for id in ids)

    with patch("automl_common.ensemble.weighted_ensemble.weighted_sum") as mock:
        ensemble.predict(x)

        args, kwargs = mock.call_args
        assert list(weights) == list(args[0])

        for passed_pred, expected_pred in zip(args[1], expected_predictions):
            np.testing.assert_equal(passed_pred, expected_pred)
