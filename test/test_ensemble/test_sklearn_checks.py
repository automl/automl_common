"""Checks sklearn compatibility

# https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/utils/estimator_checks.py#L514

Parameters
----------
ensemble : Ensemble
The ensemble to check
"""  # noqa: E501
try:
    from sklearn.utils.estimator_checks import check_estimator
except ModuleNotFoundError:
    check_estimator = None


import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.sklearn.ensemble import Ensemble

import test.test_sklearn.test_ensemble.cases as cases


@pytest.mark.skipif(check_estimator is None, reason="Needs sklearn installed")
@parametrize_with_cases("ensemble", cases=cases, filter=~ft.has_tag("fitted"))
def test_sklearn_compatibility(ensemble: Ensemble) -> None:
    checks = check_estimator(ensemble, generate_only=True)
    for estimator, check in checks:
        check(estimator)
