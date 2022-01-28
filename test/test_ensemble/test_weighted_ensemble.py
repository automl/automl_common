from unittest.mock import patch

import numpy as np
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.ensemble import WeightedEnsemble

import test.test_ensemble.cases as cases


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("valid"),
)
def test_weights_property(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble: WeightedEnsemble
        A weighted ensemble with weights

    Expects
    -------
    * The weights should be in the same order as the weighted identifiers
    """
    weights = ensemble.weights
    assert weights == list(ensemble.weighted_identifiers.values())


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("valid"),
)
def test_weight(ensemble: WeightedEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble: WeightedEnsemble
        A weighted ensemble with weights

    Expects
    -------
    * The weight function should give the same weights as weighted_identifiers
    """
    for id in ensemble.models:
        assert ensemble.weight(id) == ensemble.weighted_identifiers[id]


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("weighted") & ft.has_tag("valid"),
)
def test_predict(ensemble: WeightedEnsemble):
    """
    Parameters
    ----------
    ensemble: WeightedEnsemble
        A weighted ensemble with weights

    Expects
    -------
    * weighted_sum function should recieve the weights in the correct order with the
        predictions.
    """
    x = np.asarray([1, 10, 100])

    ids, weights = zip(*ensemble.weighted_identifiers.items())
    expected_predictions = iter(ensemble.models[id].load().predict(x) for id in ids)

    with patch("automl_common.ensemble.weighted_ensemble.weighted_sum") as mock:
        ensemble.predict(x)

        args, kwargs = mock.call_args
        assert list(weights) == args[0]

        for passed_pred, expected_pred in zip(args[1], expected_predictions):
            np.testing.assert_equal(passed_pred, expected_pred)
