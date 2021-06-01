import os
import shutil
import sys
import time

import dask.distributed

import numpy as np

import pandas as pd

import pytest

from smac.runhistory.runhistory import RunHistory, RunValue, RunKey

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from test_ensemble.ensemble_utils import (
    BackendMock,
)  # noqa (E402: module level import not   at top of file)


@pytest.fixture(scope="function")
def ensemble_backend(request):
    test_id = "%s_%s" % (request.module.__name__, request.node.name)
    test_dir = os.path.join(this_directory, test_id)

    try:
        shutil.rmtree(test_dir)
    except:  # noqa E722
        pass

    # Make sure the folders we wanna create do not already exist.
    backend = BackendMock(test_dir)

    def get_finalizer(ensemble_backend):
        def session_run_at_end():
            try:
                shutil.rmtree(test_dir)
            except:  # noqa E722
                pass

        return session_run_at_end

    request.addfinalizer(get_finalizer(backend))

    return backend


@pytest.fixture(scope="function")
def ensemble_run_history(request):

    run_history = RunHistory()
    run_history._add(
        RunKey(config_id=3, instance_id='{"task_id": "breast_cancer"}', seed=1, budget=3.0),
        RunValue(
            cost=0.11347517730496459,
            time=0.21858787536621094,
            status=None,
            starttime=time.time(),
            endtime=time.time(),
            additional_info={
                "duration": 0.20323538780212402,
                "num_run": 3,
                "configuration_origin": "Random Search",
            },
        ),
        status=None,
        origin=None,
    )
    run_history._add(
        RunKey(config_id=6, instance_id='{"task_id": "breast_cancer"}', seed=1, budget=6.0),
        RunValue(
            cost=2 * 0.11347517730496459,
            time=2 * 0.21858787536621094,
            status=None,
            starttime=time.time(),
            endtime=time.time(),
            additional_info={
                "duration": 0.20323538780212402,
                "num_run": 6,
                "configuration_origin": "Random Search",
            },
        ),
        status=None,
        origin=None,
    )
    return run_history


@pytest.fixture(scope="function")
def dask_client_single_worker(request):
    client = dask.distributed.Client(n_workers=1, threads_per_worker=1, processes=False)
    print("Started Dask client={}\n".format(client))

    def get_finalizer(address):
        def session_run_at_end():
            client = dask.distributed.get_client(address)
            print("Closed Dask client={}\n".format(client))
            client.shutdown()
            client.close()
            del client

        return session_run_at_end

    request.addfinalizer(get_finalizer(client.scheduler_info()["address"]))

    return client
