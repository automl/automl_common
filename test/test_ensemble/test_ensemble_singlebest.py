import os
import sys
import unittest.mock

import pytest

from common.ensemble_building.singlebest_ensemble import SingleBest

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from ensemble_utils import accuracy_loss_fn, log_loss_fn  # noqa (E402: module level import not   at top of file)


@unittest.mock.patch('os.path.exists')
def test_get_identifiers_from_run_history(exists, ensemble_run_history, ensemble_backend):
    exists.return_value = True
    ensemble = SingleBest(
        seed=1,
        run_history=ensemble_run_history,
        backend=ensemble_backend,
    )

    # Just one model
    assert len(ensemble.identifiers_) == 1

    # That model must be the best
    seed, num_run, budget = ensemble.identifiers_[0]
    assert num_run == 3
    assert seed == 1
    assert budget == 3.0
