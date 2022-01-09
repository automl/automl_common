from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from pathlib import Path

import dask
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.abstract_racer import AbstractRacer
from smac.intensification.intensification import Intensifier, SimpleIntensifier
from smac.runhistory.runhistory2epm import (
    AbstractRunHistory2EPM,
    RunHistory2EPM4LogCost,
)
from smac.scenario.scenario import Scenario

from automl_common.backend import Backend
from automl_common.optimizer import Optimizer
from automl_common.optimizer.smac.util import RunCompleteWrap


class SMACOptimizer(Optimizer):

    _default_scenario_args: Dict[str, Any] = {
        "save-results-instantly": True,
        "run_ob": "quality",
    }

    def __init__(
        self, backend: Backend, dask_client: Optional[dask.distributed.Client] = None
    ):
        """
        Parameters
        ----------
        backend: Backend
            The backend to use
        """
        super().__init__(backend)
        self.dask_client = dask_client

    def optimize(
        self,
        fn: Callable[..., float],
        memory: int,
        config_space: ConfigurationSpace,
        same_seed: bool = True,
        abort_on_crash: bool = False,
        n_jobs: int = 1,
        cost_for_crash: Optional[float] = None,
        max_calls: Optional[int] = None,
        max_call_time: Optional[int] = None,
        total_time: Optional[int] = None,
        instances: Optional[List[Any]] = None,
        initial_configurations: Optional[Sequence[Configuration]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        intensifier: Optional[AbstractRacer] = None,
        runhistory2epm: AbstractRunHistory2EPM = RunHistory2EPM4LogCost,
    ) -> SMAC4AC:
        """TODO"""
        # Populate the config with any parameter which was not None
        config = {
            "deterministic": same_seed,
            "abort_on_first_crash": abort_on_crash,
            "cost_for_crash": cost_for_crash,
            "runcount_limit": max_calls,
            "cutoff": max_call_time,
            "wallclock_limit": total_time,
            "memory_limit": memory,
            "instances": instances,
            "output_dir": str(self.output_dir),
            **self._default_scenario_args,
        }
        config = {k: v for k, v in config.items() if v is not None}

        if instances is not None and len(instances) > 1:
            intensifier = Intensifier
        else:
            intensifier = SimpleIntensifier

        scenario = Scenario(config)
        smac = SMAC4AC(
            scenario=scenario,
            rng=random_state,
            tae_runner=fn,
            dask_client=self.dask_client,
            runhistory2epm=runhistory2epm,
            initial_configurations=initial_configurations,
            n_jobs=n_jobs,
            intensifier=intensifier,
        )

        for handler in self.handlers["run_complete"]:
            smac.register_callback(RunCompleteWrap(handler))

        smac.optimize()

        return smac

    @property
    def output_dir(self) -> Path:
        """Where the output of the optimizer will be stored"""
        return self.backend.path / "optimizer" / "smac"

    @property
    def capabilities(self) -> List[str]:
        """No listed capabilities"""
        return ["interuptable"]

    @property
    def events(self) -> List[str]:
        """No listed events"""
        return ["run_complete"]
