from typing import Callable

from math import isclose
from pathlib import Path

from pytest_cases import parametrize, parametrize_with_cases

from automl_common.ensemble import UniformEnsemble


@parametrize("n", [1, 3, 10])
def case_uniform_model(
    path: Path,
    n: int,
    make_uniform_ensemble: Callable,
    make_model: Callable,
) -> UniformEnsemble:
    """UniformEnsemble with {1,3,10} models stored"""
    models = {str(i): make_model() for i in range(n)}
    return make_uniform_ensemble(path, models)


@parametrize_with_cases("uniform_ensemble", cases=".")
def test_uniform_ensemble_distributes_weight(uniform_ensemble: UniformEnsemble) -> None:
    """
    Parameters
    ----------
    uniform_ensemble: UniformEnsemble
        UniformEnsemble with n saved model

    Expects
    -------
    * Should have an equal weight across all models
    * Weights should add up to 1
    * Each weight should be 1/n
    """
    weights = list(uniform_ensemble.weights.values())

    assert all(w == weights[0] for w in weights)

    assert isclose(sum(weights), 1)

    expected = 1.0 / len(weights)
    assert all(isclose(w, expected) for w in weights)
