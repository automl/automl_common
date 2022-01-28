from typing import Any, Dict, Type

import numpy as np
import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from automl_common.ensemble import Ensemble

import test.test_ensemble.cases as cases


@parametrize_with_cases(
    "cls, args, err, msg",
    cases=cases,
    filter=ft.has_tag("invalid") & ft.has_tag("params"),
)
def test_construction_validation(
    cls: Type[Ensemble],
    args: Dict[str, Any],
    err: Type[Exception],
    msg: str,
) -> None:
    """
    Parameters
    ----------
    cls: Type[Ensemble]
        The Ensemble class to construct

    args: Dict[str, Any]
        The arguments to construct with

    err: Exception
        The error to catch

    msg: str
        The msg to check for

    Expects
    -------
    * The construction should fail with the given args
    """
    with pytest.raises(err, match=msg):
        cls(**args)


@parametrize_with_cases("ensemble", cases=cases, filter=ft.has_tag("valid"))
def test_model_access(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        Ensemble with saved models and corresponding identifiers

    Expects
    -------
    * Ensemble has same identifiers as its `models` property, same order
    * Can load each model in its identifiers
    """
    assert ensemble.identifiers == list(ensemble.models)
    assert all(ensemble.models[id].load() is not None for id in ensemble.identifiers)


@parametrize_with_cases("ensemble", cases=cases, filter=ft.has_tag("valid"))
def test_predict_with_models(ensemble: Ensemble) -> None:
    """
    Parameters
    ----------
    ensemble: Ensemble
        Ensemble with saved models that can be loaded to predict

    Expects
    -------
    * Ensemble be able to predict
    """
    x = np.array([0])
    ensemble.predict(x)
