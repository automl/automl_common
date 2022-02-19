from automl_common.sklearn.ensemble import SingleEnsemble

import pytest
import test.test_sklearn.test_ensemble.cases as cases
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases


@parametrize_with_cases(
    "ensemble",
    cases=cases,
    filter=~ft.has_tag("fitted") & ft.has_tag("single"),
)
def test_single_ensemble_id_raises_when_not_fitted(ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    ensemble : SingleEnsemble
        A singele ensemble that has not been fitted

    Expects
    -------
    * Accessing the id of a non-fitted SingleEnsemble should raise an error
    """
    with pytest.raises(AttributeError):
        ensemble.id

    return  # pragma: no cover
