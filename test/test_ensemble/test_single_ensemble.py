from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.ensemble import SingleEnsemble

import test.test_ensemble.cases as cases


@parametrize_with_cases(
    "single_ensemble",
    cases=cases,
    filter=ft.has_tag("single") & ft.has_tag("valid"),
)
def test_single_ensemble_model(single_ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    single_ensemble: SingleEnsemble
        SingleEnsemble with a saved model

    Expects
    -------
    * Should be able to access single model
    """
    assert single_ensemble.model.load() is not None


@parametrize_with_cases(
    "single_ensemble",
    cases=cases,
    filter=ft.has_tag("single") & ft.has_tag("valid"),
)
def test_single_ensemble_weight(single_ensemble: SingleEnsemble) -> None:
    """
    Parameters
    ----------
    single_ensemble: SingleEnsemble
        SingleEnsemble with a saved model

    Expects
    -------
    * Should have a weight of one on its single model
    """
    assert single_ensemble.model.load() is not None

    id = single_ensemble.model.id
    assert single_ensemble.models[id] == single_ensemble.model
