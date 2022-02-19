import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from sklearn.exceptions import NotFittedError

from automl_common.sklearn.ensemble import SingleEnsemble

from test.test_sklearn.test_ensemble.cases import cases


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("single"),
)
def test_properties_raise_when_not_fitted(ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : SingleEnsemble
        A singele ensemble that has not been fitted

    Expects
    -------
    * Accessing the each of properties ["id", "model"] should raise NotFittedError
    """
    properties = ["id", "model"]
    for property in properties:
        with pytest.raises(NotFittedError):
            getattr(ensemble, property)

    return  # pragma: no cover


@parametrize_with_cases("ensemble", cases=cases, has_tag=["fitted", "single"])
def test_id_is_in_ensemble(ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : SingleEnsemble
        A singele ensemble that has not been fitted

    Expects
    -------
    * Should be able to access the id it advertises
    """
    assert ensemble[ensemble.id] is not None
