from typing import TypeVar

from automl_common.backend.accessors.ensemble_accessor import EnsembleAccessor
from automl_common.ensemble import Ensemble

import test.test_backend.test_accessors.cases as cases
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

ET = TypeVar("ET", bound=Ensemble)


@parametrize_with_cases("accessor", cases=cases, filter=ft.has_tag("predictions"))
def test_predictions_with_predictions(accessor: EnsembleAccessor[ET]) -> None:
    """
    accessor: Accessor
        An EnsembleAccessor with predictions stored

    Expects
    -------
    * Should have a non 0 length
    * Should contain all the predictions it iterates through
    """
    assert len(accessor.predictions) > 0
    for key in accessor.predictions:
        assert accessor.predictions[key] is not None


@parametrize_with_cases("accessor", cases=cases, filter=~ft.has_tag("predictions"))
def test_predictions_no_predictions(accessor: EnsembleAccessor[ET]) -> None:
    """
    accessor: EnsembleAccessor[ET]
        An EnsembleAccessor with no predictions stored

    Expects
    -------
    * Should have a 0 length
    * Should provide an empty iterator
    """
    assert len(accessor.predictions) == 0
    assert len(list(accessor.predictions)) == 0
