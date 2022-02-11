import numpy as np
from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble import Ensemble

import test.test_ensemble.cases as cases


@parametrize_with_cases("ensemble", cases=cases)
def test_model_access(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        Ensemble with saved models and corresponding identifiers

    Expects
    -------
    * Can load each model in its identifiers
    """
    assert all(ensemble[id] is not None for id in ensemble.ids)


@parametrize_with_cases("ensemble", cases=cases)
def test_iterator_matches_ids(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        Ensemble to test

    Expects
    -------
    * The ids property should have the same order as the iterator
    """
    assert ensemble.ids == list(ensemble)


@parametrize_with_cases("ensemble", cases=cases)
def test_length_matches_id_count(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        Ensemble to test

    Expects
    -------
    * The length of the ensenmble is defined to be how many ids it contains
    """
    assert len(ensemble) == len(ensemble.ids)


@parametrize_with_cases("ensemble", cases=cases)
def test_contains_all_ids(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        An Ensemble to test

    Expects
    -------
    * Ensemble should contain every id it has in it's ids property
    """
    assert all(id in ensemble for id in ensemble.ids)


@parametrize_with_cases("ensemble", cases=cases)
@parametrize("x", [np.array([1, 1, 1]), np.array([[1, 1, 1], [1, 1, 1]])])
def test_takes_predictions(ensemble: Ensemble, x: np.ndarray) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        An Ensemble to test

    x: np.ndarray
        The values to predict on

    Expects
    -------
    * Ensemble should be able to produce predictions
    """
    assert ensemble.predict(x) is not None
