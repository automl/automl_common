from typing import Callable, Tuple

from unittest.mock import patch

import numpy as np
import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases
from sklearn.exceptions import NotFittedError

from automl_common.sklearn.ensemble import ClassifierEnsemble

import test.test_sklearn.test_ensemble.test_classification.cases as cases

DataFactory = Callable[..., Tuple[np.ndarray, np.ndarray]]


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_predict_proba_raises_when_not_fitted(ensemble: ClassifierEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : ClassifierEnsemble
        An unfitted ClassifierEnsemble

    Expects
    -------
    * The unfitted ensemble should not be able to predict_proba
    """
    with pytest.raises(NotFittedError):
        ensemble.predict_proba(np.array([1, 2, 3, 4]))

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_predict_proba_does_not_modify_data(
    ensemble: ClassifierEnsemble,
    make_xy: DataFactory,
) -> None:
    """
    Parameters
    ----------
    ensemble : ClassifierEnsemble
        An unfitted ClassifierEnsemble

    make_xy: DataFactory
        Factory for making x, y data

    Expects
    -------
    * The ensemble predict_proba should not modify data
    """
    x, y = make_xy(kind="classification")
    x_ = x.copy()

    ensemble.fit(x, y)
    ensemble.predict_proba(x)

    np.testing.assert_equal(x, x_)


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=ft.has_tag("classifier") & ft.has_tag("fitted"),
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_proba_raises_with_jagged_predictions(ensemble: ClassifierEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : Ensemble
        A fitted classifier ensemble

    Expects
    -------
    * A runtime error should be raise as the probability predictions don't have the
      correct size
    """
    jagged = np.array([[1, 2, 3], [1, 2]])
    with patch.object(ensemble, "_predict_proba", return_value=jagged):

        with pytest.raises(RuntimeError, match="Probability predictions were jagged"):
            x = np.array([[1, 2, 3], [1, 2, 3]])
            ensemble.predict_proba(x)

    return  # pragma: no cover
