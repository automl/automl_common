# -*- encoding: utf-8 -*-
import glob
import gzip
import logging.handlers
import math
import multiprocessing
import os
import pickle
import re
import shutil
import time
import traceback
import zlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import dask.distributed

import numpy as np

import pandas as pd

import pynisher

from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_random_state

from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae.base import StatusType

from .abstract_ensemble import AbstractEnsemble
from .ensemble_selection import EnsembleSelection
from ..utils.backend import Backend
from ..utils.logging_ import get_named_client_logger
from ..utils.parallel import preload_modules

Y_ENSEMBLE = 0
Y_VALID = 1
Y_TEST = 2

MODEL_FN_RE = r"_([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*)\.npy"


# TODO: SMAC IncorporateRunResultCallback need to be typed properly
# to prevent: error: Class cannot subclass 'IncorporateRunResultCallback' (has type 'Any')  [misc]
class EnsembleBuilderManager(IncorporateRunResultCallback):  # type: ignore
    def __init__(
        self,
        start_time: float,
        time_left_for_ensembles: float,
        backend: Backend,
        dataset_name: str,
        ensemble_size: int,
        ensemble_nbest: Union[int, float],
        max_models_on_disc: Optional[Union[float, int]],
        seed: int,
        precision: int,
        max_iterations: Optional[int],
        read_at_most: int,
        ensemble_memory_limit: Optional[int],
        random_state: int,
        loss_fn: Callable[..., float],
        loss_fn_args: Dict[str, Any],
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        pynisher_context: str = "fork",
        preload_modules_list: Optional[List[str]] = None,
    ):
        """SMAC callback to handle ensemble building

            Parameters
            ----------
            start_time: int
                the time when this job was started, to account for any latency in job allocation
            time_left_for_ensemble: int
                How much time is left for the task. Job should finish within this allocated time
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            ensemble_size: int
                maximal size of ensemble
            ensemble_nbest: int/float
                if int: consider only the n best prediction
                if float: consider only this fraction of the best models
                Both wrt to validation predictions
                If performance_range_threshold > 0, might return less models
            max_models_on_disc: int
               Defines the maximum number of models that are kept in the disc.
               If int, it must be greater or equal than 1, and dictates the max number of
               models to keep.
               If float, it will be interpreted as the max megabytes allowed of disc space. That
               is, if the number of ensemble candidates require more disc space than this float
               value, the worst models will be deleted to keep within this budget.
               Models and predictions of the worst-performing models will be deleted then.
               If None, the feature is disabled.
               It defines an upper bound on the models that can be used in the ensemble.
            seed: int
                random seed
            max_iterations: int
                maximal number of iterations to run this script
                (default None --> deactivated)
            precision: [16,32,64,128]
                precision of floats to read the predictions
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
            read_at_most: int
                read at most n new prediction files in each iteration
            loss_fn: Callable[..., float]
                A function to calculate loss between a set of labels and predictions from the
                ensemble
            loss_fn_args: Dict[str, Any]
                Arguments that are passed to the loss function
            logger_port: int
                port that receives logging records
            pynisher_context: str
                The multiprocessing context for pynisher. One of spawn/fork/forkserver.
            preload_modules_list: str
                The list of modules to pre-load. This makes sense only when the context is
                forkserver.

        Returns
        -------
            List[Tuple[int, float, float, float]]:
                A list with the performance history of this ensemble, of the form
                [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
        """
        self.start_time = start_time
        self.time_left_for_ensembles = time_left_for_ensembles
        self.backend = backend
        self.dataset_name = dataset_name
        self.loss_fn = loss_fn
        self.loss_fn_args = loss_fn_args
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc = max_models_on_disc
        self.seed = seed
        self.precision = precision
        self.max_iterations = max_iterations
        self.read_at_most = read_at_most
        self.ensemble_memory_limit = ensemble_memory_limit
        self.random_state = random_state
        self.logger_port = logger_port
        self.pynisher_context = pynisher_context
        self.preload_modules_list = preload_modules_list

        # Store something similar to SMAC's runhistory
        self.history: List[Dict[str, float]] = []

        # We only submit new ensembles when there is not an active ensemble job
        self.futures: List[dask.distributed.Future] = []

        # The last criteria is the number of iterations
        self.iteration = 0

        # Keep track of when we started to know when we need to finish!
        self.start_time = time.time()

    def __call__(
        self,
        smbo: "SMBO",
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> None:
        if result.status in (StatusType.STOP, StatusType.ABORT) or smbo._stop:
            return
        self.build_ensemble(smbo.tae_runner.client)

    def build_ensemble(self, dask_client: dask.distributed.Client, unit_test: bool = False) -> None:

        # The second criteria is elapsed time
        elapsed_time = time.time() - self.start_time

        logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        # First test for termination conditions
        if self.time_left_for_ensembles < elapsed_time:
            logger.info(
                "Terminate ensemble building as not time is left (run for {}s)".format(
                    elapsed_time
                ),
            )
            return
        if self.max_iterations is not None and self.max_iterations <= self.iteration:
            logger.info(
                "Terminate ensemble building because of max iterations: {} of {}".format(
                    self.max_iterations, self.iteration
                )
            )
            return

        if len(self.futures) != 0:
            if self.futures[0].done():
                result = self.futures.pop().result()
                if result:
                    ensemble_history, self.ensemble_nbest, _, _, _ = result
                    logger.debug(
                        "iteration={} @ elapsed_time={} has history={}".format(
                            self.iteration,
                            elapsed_time,
                            ensemble_history,
                        )
                    )
                    self.history.extend(ensemble_history)

        # Only submit new jobs if the previous ensemble job finished
        if len(self.futures) == 0:

            # Add the result of the run
            # On the next while iteration, no references to
            # ensemble builder object, so it should be garbage collected to
            # save memory while waiting for resources
            # Also, notice how ensemble nbest is returned, so we don't waste
            # iterations testing if the deterministic predictions size can
            # be fitted in memory
            try:
                # Submit a Dask job from this job, to properly
                # see it in the dask diagnostic dashboard
                # Notice that the forked ensemble_builder_process will
                # wait for the below function to be done
                self.futures.append(
                    dask_client.submit(
                        fit_and_return_ensemble,
                        backend=self.backend,
                        dataset_name=self.dataset_name,
                        loss_fn=self.loss_fn,
                        loss_fn_args=self.loss_fn_args,
                        ensemble_size=self.ensemble_size,
                        ensemble_nbest=self.ensemble_nbest,
                        max_models_on_disc=self.max_models_on_disc,
                        seed=self.seed,
                        precision=self.precision,
                        memory_limit=self.ensemble_memory_limit,
                        read_at_most=self.read_at_most,
                        random_state=self.seed,
                        end_at=self.start_time + self.time_left_for_ensembles,
                        iteration=self.iteration,
                        return_predictions=False,
                        priority=100,
                        pynisher_context=self.pynisher_context,
                        preload_modules_list=self.preload_modules_list,
                        logger_port=self.logger_port,
                        unit_test=unit_test,
                    )
                )

                logger.info(
                    "{}/{} Started Ensemble builder job at {} for iteration {}.".format(
                        # Log the client to make sure we
                        # remain connected to the scheduler
                        self.futures[0],
                        dask_client,
                        time.strftime("%Y.%m.%d-%H.%M.%S"),
                        self.iteration,
                    ),
                )
                self.iteration += 1
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                logger.critical(exception_traceback)
                logger.critical(error_message)


def fit_and_return_ensemble(
    backend: Backend,
    dataset_name: str,
    loss_fn: Callable[..., float],
    loss_fn_args: Dict[str, Any],
    ensemble_size: int,
    ensemble_nbest: Union[int, float],
    max_models_on_disc: Optional[Union[float, int]],
    seed: int,
    precision: int,
    memory_limit: Optional[int],
    read_at_most: int,
    random_state: int,
    end_at: float,
    iteration: int,
    return_predictions: bool,
    pynisher_context: str,
    preload_modules_list: Optional[List[str]] = None,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    unit_test: bool = False,
) -> Optional[
    Tuple[
        List[Dict[str, float]],
        Union[float, int],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]
]:
    """

    A short function to fit and create an ensemble. It is just a wrapper to easily send
    a request to dask to create an ensemble and clean the memory when finished

    Parameters
    ----------
        backend: util.backend.Backend
            backend to write and read files
        dataset_name: str
            name of dataset
        loss_fn: Callable[..., float]
            A loss function to calculate the error between the ensemble predictions
            and the ground truth
        loss_fn_args: Dict[str, Any]
            A set of arguments passed to the loss_fn
        ensemble_size: int
            maximal size of ensemble
        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models
        max_models_on_disc: int
           Defines the maximum number of models that are kept in the disc.
           If int, it must be greater or equal than 1, and dictates the max number of
           models to keep.
           If float, it will be interpreted as the max megabytes allowed of disc space. That
           is, if the number of ensemble candidates require more disc space than this float
           value, the worst models will be deleted to keep within this budget.
           Models and predictions of the worst-performing models will be deleted then.
           If None, the feature is disabled.
           It defines an upper bound on the models that can be used in the ensemble.
        seed: int
            random seed
        precision: [16,32,64,128]
            precision of floats to read the predictions
        memory_limit: Optional[int]
            memory limit in mb. If ``None``, no memory limit is enforced.
        read_at_most: int
            read at most n new prediction files in each iteration
        end_at: float
            At what time the job must finish. Needs to be the endtime and not the time left
            because we do not know when dask schedules the job.
        iteration: int
            The current iteration
        pynisher_context: str
            Context to use for multiprocessing, can be either fork, spawn or forkserver.
        preload_modules_list: List[str]
            A list of modules to pre-load in the context of forkserver
        logger_port: int
            The port where the logging server is listening to.
        unit_test: bool
            Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
            Having this is very bad coding style, but I did not find a way to make
            unittest.mock work through the pynisher with all spawn contexts. If you know a
            better solution, please let us know by opening an issue.

    Returns
    -------
        List[Tuple[int, float, float, float]]
            A list with the performance history of this ensemble, of the form
            [[pandas_timestamp, train_performance, val_performance, test_performance], ...]

    """
    result = EnsembleBuilder(
        backend=backend,
        dataset_name=dataset_name,
        loss_fn=loss_fn,
        loss_fn_args=loss_fn_args,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
        seed=seed,
        precision=precision,
        memory_limit=memory_limit,
        read_at_most=read_at_most,
        random_state=random_state,
        logger_port=logger_port,
        preload_modules_list=preload_modules_list,
        unit_test=unit_test,
    ).run(
        end_at=end_at,
        iteration=iteration,
        return_predictions=return_predictions,
        pynisher_context=pynisher_context,
    )
    return result


class EnsembleBuilder(object):
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        loss_fn: Callable[..., float],
        loss_fn_args: Dict[str, Any],
        ensemble_size: int = 10,
        ensemble_nbest: Union[int, float] = 100,
        max_models_on_disc: Optional[Union[float, int]] = 100,
        performance_range_threshold: float = 0,
        seed: int = 1,
        precision: int = 32,
        memory_limit: Optional[int] = 1024,
        read_at_most: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        preload_modules_list: Optional[List[str]] = None,
        unit_test: bool = False,
    ):
        """
        Constructor

        Parameters
        ----------
        backend: util.backend.Backend
            backend to write and read files
        dataset_name: str
            name of dataset
        loss_fn: Callable[..., float]
            A loss function to calculate the error between the ensemble predictions
            and the ground truth
        loss_fn_args: Dict[str, Any]
            A set of arguments passed to the loss_fn
        ensemble_size: int
            maximal size of ensemble
        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models
        max_models_on_disc: int
           Defines the maximum number of models that are kept in the disc.
           If int, it must be greater or equal than 1, and dictates the max number of
           models to keep.
           If float, it will be interpreted as the max megabytes allowed of disc space. That
           is, if the number of ensemble candidates require more disc space than this float
           value, the worst models will be deleted to keep within this budget.
           Models and predictions of the worst-performing models will be deleted then.
           If None, the feature is disabled.
           It defines an upper bound on the models that can be used in the ensemble.
        performance_range_threshold: float
            Keep only models that are better than:
                dummy + (best - dummy)*performance_range_threshold
            E.g dummy=2, best=4, thresh=0.5 --> only consider models with loss > 3
            Will at most return the minimum between ensemble_nbest models,
            and max_models_on_disc. Might return less
        seed: int
            random seed
        precision: [16,32,64,128]
            precision of floats to read the predictions
        memory_limit: Optional[int]
            memory limit in mb. If ``None``, no memory limit is enforced.
        read_at_most: int
            read at most n new prediction files in each iteration
        logger_port: int
            port that receives logging records
        preload_modules_list: List[str]
            A list of modules to pre-load
        unit_test: bool
            Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
            Having this is very bad coding style, but I did not find a way to make
            unittest.mock work through the pynisher with all spawn contexts. If you know a
            better solution, please let us know by opening an issue.
        """

        super(EnsembleBuilder, self).__init__()

        self.backend = backend  # communication with filesystem
        self.dataset_name = dataset_name
        self.loss_fn = loss_fn
        self.loss_fn_args = loss_fn_args
        self.ensemble_size = ensemble_size
        self.performance_range_threshold = performance_range_threshold

        if isinstance(ensemble_nbest, int) and ensemble_nbest < 1:
            raise ValueError(f"Integer ensemble_nbest has to be larger 1:{ensemble_nbest}")
        elif not isinstance(ensemble_nbest, int):
            if ensemble_nbest < 0 or ensemble_nbest > 1:
                raise ValueError(
                    "Float ensemble_nbest best has to be >= 0 and <= 1: {ensemble_nbest}"
                )

        self.ensemble_nbest = ensemble_nbest

        # max_models_on_disc can be a float, in such case we need to
        # remember the user specified Megabytes and translate this to
        # max number of ensemble models. max_resident_models keeps the
        # maximum number of models in disc
        if max_models_on_disc is not None and max_models_on_disc < 0:
            raise ValueError("max_models_on_disc has to be a positive number or None")
        self.max_models_on_disc = max_models_on_disc
        self.max_resident_models: Optional[int] = None

        self.seed = seed
        self.precision = precision
        self.memory_limit = memory_limit
        self.read_at_most = read_at_most
        self.random_state = check_random_state(random_state)
        self.preload_modules_list = preload_modules_list
        self.unit_test = unit_test

        # Setup the logger
        self.logger_port = logger_port
        self.logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        if ensemble_nbest == 1:
            self.logger.debug(
                "Behaviour depends on int/float: {}, {} (ensemble_nbest, type)".format(
                    ensemble_nbest, type(ensemble_nbest)
                )
            )

        self.start_time = 0.0
        self.model_fn_re = re.compile(MODEL_FN_RE)

        self.last_hash = None  # hash of ensemble training data
        self.y_true_ensemble = None
        self.SAVE2DISC = True

        # already read prediction files
        # {"file name": {
        #    "ens_loss": float
        #    "mtime_ens": str,
        #    "mtime_valid": str,
        #    "mtime_test": str,
        #    "seed": int,
        #    "num_run": int,
        # }}
        self.read_losses = {}
        # {"file_name": {
        #    Y_ENSEMBLE: np.ndarray
        #    Y_VALID: np.ndarray
        #    Y_TEST: np.ndarray
        #    }
        # }
        self.read_preds = {}

        # Depending on the dataset dimensions,
        # regenerating every iteration, the predictions
        # losses for self.read_preds
        # is too computationally expensive
        # As the ensemble builder is stateless
        # (every time the ensemble builder gets resources
        # from dask, it builds this object from scratch)
        # we save the state of this dictionary to memory
        # and read it if available
        self.ensemble_memory_file = os.path.join(
            self.backend.internals_directory, "ensemble_read_preds.pkl"
        )
        if os.path.exists(self.ensemble_memory_file):
            try:
                with (open(self.ensemble_memory_file, "rb")) as memory:
                    self.read_preds, self.last_hash = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    "Could not load the previous iterations of ensemble_builder predictions."
                    "This might impact the quality of the run. Exception={} {}".format(
                        e,
                        traceback.format_exc(),
                    )
                )
        self.ensemble_loss_file = os.path.join(
            self.backend.internals_directory, "ensemble_read_losses.pkl"
        )
        if os.path.exists(self.ensemble_loss_file):
            try:
                with (open(self.ensemble_loss_file, "rb")) as memory:
                    self.read_losses = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    "Could not load the previous iterations of ensemble_builder losses."
                    "This might impact the quality of the run. Exception={} {}".format(
                        e,
                        traceback.format_exc(),
                    )
                )

        # hidden feature which can be activated via an environment variable. This keeps all
        # models and predictions which have ever been a candidate. This is necessary to post-hoc
        # compute the whole ensemble building trajectory.
        self._has_been_candidate: Set[str] = set()

        self.validation_performance_ = np.inf

        # Track the ensemble performance
        datamanager = self.backend.load_datamanager()  # type: ignore[var-annotated]
        self.y_valid = datamanager.data.get("Y_valid")
        self.y_test = datamanager.data.get("Y_test")
        del datamanager
        self.ensemble_history: List[Dict[str, float]] = []

    def run(
        self,
        iteration: int,
        pynisher_context: str,
        time_left: Optional[float] = None,
        end_at: Optional[float] = None,
        time_buffer: int = 5,
        return_predictions: bool = False,
    ) -> Optional[
        Tuple[
            List[Dict[str, float]],
            Union[float, int],
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ]
    ]:

        if time_left is None and end_at is None:
            raise ValueError("Must provide either time_left or end_at.")
        elif time_left is not None and end_at is not None:
            raise ValueError("Cannot provide both time_left and end_at.")

        self.logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        process_start_time = time.time()
        while True:

            if time_left is not None:
                time_elapsed = time.time() - process_start_time
                time_left -= time_elapsed
            elif end_at is not None:
                current_time = time.time()
                if current_time > end_at:
                    break
                else:
                    time_left = end_at - current_time
            else:
                raise ValueError(f"time_left={time_left} end_at={end_at}")

            wall_time_in_s = int(time_left - time_buffer)
            if wall_time_in_s < 1:
                break
            context = multiprocessing.get_context(pynisher_context)
            if self.preload_modules_list:
                preload_modules(context, self.preload_modules_list)

            safe_ensemble_script = pynisher.enforce_limits(
                wall_time_in_s=wall_time_in_s,
                mem_in_mb=self.memory_limit,
                logger=self.logger,
                context=context,
            )(self.main)
            safe_ensemble_script(time_left, iteration, return_predictions)
            if safe_ensemble_script.exit_status is pynisher.MemorylimitException:
                # if ensemble script died because of memory error,
                # reduce nbest to reduce memory consumption and try it again

                # ATTENTION: main will start from scratch; # all data structures are empty again
                try:
                    os.remove(self.ensemble_memory_file)
                except:  # noqa E722
                    pass

                if isinstance(self.ensemble_nbest, int) and self.ensemble_nbest <= 1:
                    if self.read_at_most == 1:
                        self.logger.error(
                            "Memory Exception -- Unable to further reduce the number of ensemble "
                            "members and can no further limit the number of ensemble members "
                            "loaded per iteration -- please restart the estimator with a higher "
                            "value for the argument `memory_limit` "
                            f"(current limit is {self.memory_limit} MB). "
                            "The ensemble builder will keep running to delete files from disk in "
                            "case this was enabled."
                        )
                        self.ensemble_nbest = 0
                    else:
                        self.read_at_most = 1
                        self.logger.warning(
                            "Memory Exception -- Unable to further reduce the number of ensemble "
                            "members -- Now reducing the number of predictions per call to read "
                            "at most to 1."
                        )
                else:
                    if isinstance(self.ensemble_nbest, int):
                        self.ensemble_nbest = max(1, int(self.ensemble_nbest / 2))
                    else:
                        self.ensemble_nbest = self.ensemble_nbest / 2
                    self.logger.warning(
                        "Memory Exception -- restart with "
                        f"less ensemble_nbest: {self.ensemble_nbest}"
                    )
                    return [], self.ensemble_nbest, None, None, None
            else:
                if safe_ensemble_script.result is None:
                    return safe_ensemble_script.result
                else:
                    return cast(
                        Tuple[
                            List[Dict[str, float]],
                            Union[int, float],
                            Optional[np.ndarray],
                            Optional[np.ndarray],
                            Optional[np.ndarray],
                        ],
                        safe_ensemble_script.result,
                    )

        return [], self.ensemble_nbest, None, None, None

    def main(
        self, time_left: float, iteration: int, return_predictions: bool
    ) -> Tuple[
        List[Dict[str, float]],
        Union[int, float],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:

        # Pynisher jobs inside dask 'forget'
        # the logger configuration. So we have to set it up
        # accordingly
        self.logger = get_named_client_logger(
            name="EnsembleBuilder",
            port=self.logger_port,
        )

        self.start_time = time.time()
        train_pred, valid_pred, test_pred = None, None, None

        used_time = time.time() - self.start_time
        self.logger.debug(
            "Starting iteration {}, time left: {}".format(
                iteration,
                time_left - used_time,
            )
        )

        # populates self.read_preds and self.read_losses
        if not self.compute_loss_per_model():
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, valid_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None, None

        # Only the models with the n_best predictions are candidates
        # to be in the ensemble
        candidate_models = self.get_n_best_preds()
        if not candidate_models:  # no candidates yet
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, valid_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None, None

        # populates predictions in self.read_preds
        # reduces selected models if file reading failed
        n_sel_valid, n_sel_test = self.get_valid_test_preds(selected_keys=candidate_models)

        # If valid/test predictions loaded, then reduce candidate models to this set
        if (
            len(n_sel_test) != 0
            and len(n_sel_valid) != 0
            and len(set(n_sel_valid).intersection(set(n_sel_test))) == 0
        ):
            # Both n_sel_* have entries, but there is no overlap, this is critical
            self.logger.error("n_sel_valid and n_sel_test are not empty, but do not overlap")
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, valid_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None, None

        # If any of n_sel_* is not empty and overlaps with candidate_models,
        # then ensure candidate_models AND n_sel_test are sorted the same
        candidate_models_set = set(candidate_models)
        if candidate_models_set.intersection(n_sel_valid).intersection(n_sel_test):
            candidate_models = sorted(
                candidate_models_set.intersection(n_sel_valid).intersection(n_sel_test)
            )
            n_sel_test = candidate_models
            n_sel_valid = candidate_models
        elif candidate_models_set.intersection(n_sel_valid):
            candidate_models = sorted(candidate_models_set.intersection(n_sel_valid))
            n_sel_valid = candidate_models
        elif candidate_models_set.intersection(n_sel_test):
            candidate_models = sorted(candidate_models_set.intersection(n_sel_test))
            n_sel_test = candidate_models
        else:
            # This has to be the case
            n_sel_test = []
            n_sel_valid = []

        if os.environ.get("ENSEMBLE_KEEP_ALL_CANDIDATES"):
            for candidate in candidate_models:
                self._has_been_candidate.add(candidate)

        # train ensemble
        ensemble = self.fit_ensemble(selected_keys=candidate_models)

        # Save the ensemble for later use in the main module!
        if ensemble is not None and self.SAVE2DISC:
            self.backend.save_ensemble(ensemble, iteration, self.seed)

        # Delete files of non-candidate models - can only be done after fitting the ensemble and
        # saving it to disc so we do not accidentally delete models in the previous ensemble
        if self.max_resident_models is not None:
            self._delete_excess_models(selected_keys=candidate_models)

        # Save the read losses status for the next iteration
        with open(self.ensemble_loss_file, "wb") as memory:
            pickle.dump(self.read_losses, memory)

        if ensemble is not None:
            train_pred = self.predict(
                set_="train",
                ensemble=ensemble,
                selected_keys=candidate_models,
                n_preds=len(candidate_models),
                index_run=iteration,
            )
            # We can't use candidate_models here, as n_sel_* might be empty
            valid_pred = self.predict(
                set_="valid",
                ensemble=ensemble,
                selected_keys=n_sel_valid,
                n_preds=len(candidate_models),
                index_run=iteration,
            )
            # TODO if predictions fails, build the model again during the
            #  next iteration!
            test_pred = self.predict(
                set_="test",
                ensemble=ensemble,
                selected_keys=n_sel_test,
                n_preds=len(candidate_models),
                index_run=iteration,
            )

            # Add losses to run history to see ensemble progress
            self._add_ensemble_trajectory(train_pred, valid_pred, test_pred)

        # The loaded predictions and the hash can only be saved after the ensemble has been
        # built, because the hash is computed during the construction of the ensemble
        with open(self.ensemble_memory_file, "wb") as memory:
            pickle.dump((self.read_preds, self.last_hash), memory)

        if return_predictions:
            return self.ensemble_history, self.ensemble_nbest, train_pred, valid_pred, test_pred
        else:
            return self.ensemble_history, self.ensemble_nbest, None, None, None

    def get_disk_consumption(self, pred_path: str) -> float:
        """
        gets the cost of a model being on disc
        """

        match = self.model_fn_re.search(pred_path)
        if not match:
            raise ValueError(f"Invalid path format {pred_path}")
        _seed = int(match.group(1))
        _num_run = int(match.group(2))
        _budget = float(match.group(3))

        stored_files_for_run = os.listdir(
            self.backend.get_numrun_directory(_seed, _num_run, _budget)
        )
        stored_files_for_run = [
            os.path.join(self.backend.get_numrun_directory(_seed, _num_run, _budget), file_name)
            for file_name in stored_files_for_run
        ]
        this_model_cost = sum([os.path.getsize(path) for path in stored_files_for_run])

        # get the megabytes
        return round(this_model_cost / math.pow(1024, 2), 2)

    def compute_loss_per_model(self) -> bool:
        """
        Compute the loss of the predictions on ensemble building data set;
        populates self.read_preds and self.read_losses
        """

        self.logger.debug("Read ensemble data set predictions")

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: {}".format(
                        traceback.format_exc(),
                    ),
                )
                return False

        pred_path = os.path.join(
            glob.escape(self.backend.get_runs_directory()),
            "%d_*_*" % self.seed,
            "predictions_ensemble_%s_*_*.npy*" % self.seed,
        )
        y_ens_files = glob.glob(pred_path)
        y_ens_files = [
            y_ens_file
            for y_ens_file in y_ens_files
            if y_ens_file.endswith(".npy") or y_ens_file.endswith(".npy.gz")
        ]
        self.y_ens_files = y_ens_files
        # no validation predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug(f"Found no prediction files on ensemble data set: {pred_path}")
            return False

        # First sort files chronologically
        to_read = []
        for _y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(_y_ens_fn)
            if match is None:
                raise ValueError(f"Could not interpret y_ens_fn={_y_ens_fn}")
            seed = int(match.group(1))
            num_run = int(match.group(2))
            budget = float(match.group(3))
            mtime = os.path.getmtime(_y_ens_fn)

            to_read.append([_y_ens_fn, match, seed, num_run, budget, mtime])

        n_read_files = 0
        # Now read file wrt to num_run
        for y_ens_fn, _match, _seed, _num_run, _budget, mtime in cast(
            List[Tuple[str, Any, int, int, float, float]], sorted(to_read, key=lambda x: x[5])
        ):
            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                self.logger.info(f"Error loading file (not .npy or .npy.gz): {y_ens_fn}")
                continue

            if not self.read_losses.get(y_ens_fn):
                self.read_losses[y_ens_fn] = {
                    "ens_loss": np.inf,
                    "mtime_ens": 0,
                    "mtime_valid": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    "disc_space_cost_mb": None,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    # 3 - deleted from disk due to space constraints
                    "loaded": 0,
                }
            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    Y_ENSEMBLE: None,
                    Y_VALID: None,
                    Y_TEST: None,
                }

            if self.read_losses[y_ens_fn]["mtime_ens"] == mtime:
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._read_np_fn(y_ens_fn)
                loss = self.loss_fn(self.y_true_ensemble, y_ensemble, **self.loss_fn_args)

                if np.isfinite(self.read_losses[y_ens_fn]["ens_loss"]):
                    self.logger.debug(
                        f"Changing ensemble loss for file {y_ens_fn} from "
                        f"{self.read_losses[y_ens_fn]['ens_loss']} to {loss} "
                        "because file modification time changed? "
                        f"{self.read_losses[y_ens_fn]['mtime_ens']} "
                        f"- {os.path.getmtime(y_ens_fn)}",
                    )

                self.read_losses[y_ens_fn]["ens_loss"] = loss

                # It is not needed to create the object here
                # To save memory, we just compute the loss.
                self.read_losses[y_ens_fn]["mtime_ens"] = os.path.getmtime(y_ens_fn)
                self.read_losses[y_ens_fn]["loaded"] = 2
                self.read_losses[y_ens_fn]["disc_space_cost_mb"] = self.get_disk_consumption(
                    y_ens_fn
                )

                n_read_files += 1

            except Exception:
                self.logger.warning(
                    "Error loading {}: {}".format(
                        y_ens_fn,
                        traceback.format_exc(),
                    )
                )
                self.read_losses[y_ens_fn]["ens_loss"] = np.inf

        self.logger.debug(
            "Done reading {} new prediction files. Loaded {} predictions in total.".format(
                n_read_files,
                np.sum([pred["loaded"] > 0 for pred in self.read_losses.values()]),
            )
        )
        return True

    def get_n_best_preds(self) -> List[str]:
        """
        get best n predictions (i.e., keys of self.read_losses)
        according to the loss on the "ensemble set"
        n: self.ensemble_nbest

        Side effects:
            ->Define the n-best models to use in ensemble
            ->Only the best models are loaded
            ->Any model that is not best is candidate to deletion
              if max models in disc is exceeded.
        """

        sorted_keys = self._get_list_of_sorted_preds()

        # number of models available
        num_keys = len(sorted_keys)
        # remove all that are at most as good as random
        # note: dummy model must have run_id=1 (there is no run_id=0)
        dummy_losses = list(filter(lambda x: x[2] == 1, sorted_keys))
        # number of dummy models
        num_dummy = len(dummy_losses)
        dummy_loss = dummy_losses[0]
        self.logger.debug("Use {} as dummy loss".format(dummy_loss[1]))

        # sorted_keys looks like: (k, v["ens_loss"], v["num_run"])
        # On position 1 we have the loss of a minimization problem.
        # keep only the predictions with a loss smaller than the dummy
        # prediction
        sorted_keys = list(filter(lambda x: x[1] < dummy_loss[1], sorted_keys))

        # remove Dummy Classifier
        sorted_keys = list(filter(lambda x: x[2] > 1, sorted_keys))
        if not sorted_keys:
            # no model left; try to use dummy loss (num_run==0)
            # log warning when there are other models but not better than dummy model
            if num_keys > num_dummy:
                self.logger.warning(
                    "No models better than random - using Dummy loss!"
                    f"Number of models besides current dummy model: {num_keys - 1}. "
                    f"Number of dummy models: {num_dummy}",
                )
            sorted_keys = [
                (k, v["ens_loss"], v["num_run"])
                for k, v in self.read_losses.items()
                if v["seed"] == self.seed and v["num_run"] == 1
            ]
        # reload predictions if losses changed over time and a model is
        # considered to be in the top models again!
        if not isinstance(self.ensemble_nbest, int):
            # Transform to number of models to keep. Keep at least one
            keep_nbest = max(1, min(len(sorted_keys), int(len(sorted_keys) * self.ensemble_nbest)))
            self.logger.debug(
                f"Library pruning: using only top {self.ensemble_nbest * 100} percent of "
                f"the models for ensemble ({keep_nbest} out of {len(sorted_keys)})",
            )
        else:
            # Keep only at most ensemble_nbest
            keep_nbest = min(self.ensemble_nbest, len(sorted_keys))
            self.logger.debug(
                "Library Pruning: using for ensemble only "
                f" {keep_nbest} (out of {len(sorted_keys)}) models"
            )

        # If max_models_on_disc is None, do nothing
        # One can only read at most max_models_on_disc models
        if self.max_models_on_disc is not None:
            if not isinstance(self.max_models_on_disc, int):
                consumption = [
                    [
                        v["ens_loss"],
                        v["disc_space_cost_mb"],
                    ]
                    for v in self.read_losses.values()
                    if v["disc_space_cost_mb"] is not None
                ]
                max_consumption = max(c[1] for c in consumption)

                # We are pessimistic with the consumption limit indicated by
                # max_models_on_disc by 1 model. Such model is assumed to spend
                # max_consumption megabytes
                if (sum(c[1] for c in consumption) + max_consumption) > self.max_models_on_disc:

                    # just leave the best -- smaller is better!
                    # This list is in descending order, to preserve the best models
                    sorted_cum_consumption = (
                        np.cumsum([c[1] for c in sorted(consumption)]) + max_consumption
                    )
                    max_models = np.argmax(sorted_cum_consumption > self.max_models_on_disc)

                    # Make sure that at least 1 model survives
                    self.max_resident_models = max(1, max_models)
                    self.logger.warning(
                        "Limiting num of models via float max_models_on_disc={}"
                        " as accumulated={} worst={} num_models={}".format(
                            self.max_models_on_disc,
                            (sum(c[1] for c in consumption) + max_consumption),
                            max_consumption,
                            self.max_resident_models,
                        )
                    )
                else:
                    self.max_resident_models = None
            else:
                self.max_resident_models = self.max_models_on_disc

        if self.max_resident_models is not None and keep_nbest > self.max_resident_models:
            self.logger.debug(
                f"Restricting the number of models to {self.max_resident_models} "
                f"instead of {keep_nbest} due to argument max_models_on_disc",
            )
            keep_nbest = self.max_resident_models

        # consider performance_range_threshold
        if self.performance_range_threshold > 0:
            best_loss = sorted_keys[0][1]
            worst_loss = dummy_loss[1]
            worst_loss -= (worst_loss - best_loss) * self.performance_range_threshold
            if sorted_keys[keep_nbest - 1][1] > worst_loss:
                # We can further reduce number of models
                # since worst model is worse than thresh
                for i in range(0, keep_nbest):
                    # Look at most at keep_nbest models,
                    # but always keep at least one model
                    current_loss = sorted_keys[i][1]
                    if current_loss >= worst_loss:
                        self.logger.debug(
                            f"Dynamic Performance range: "
                            "Further reduce from {keep_nbest}"
                            f" to {max(1, i)} models",
                        )
                        keep_nbest = max(1, i)
                        break
        ensemble_n_best = keep_nbest

        # reduce to keys
        sorted_keys_str = list(map(lambda x: x[0], sorted_keys))

        # remove loaded predictions for non-winning models
        for k in sorted_keys_str[ensemble_n_best:]:
            if k in self.read_preds:
                self.read_preds[k][Y_ENSEMBLE] = None
                self.read_preds[k][Y_VALID] = None
                self.read_preds[k][Y_TEST] = None
            if self.read_losses[k]["loaded"] == 1:
                self.logger.debug(
                    "Dropping model {} ({},{}) with loss {}.".format(
                        k,
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["ens_loss"],
                    )
                )
                self.read_losses[k]["loaded"] = 2

        # Load the predictions for the winning
        for k in sorted_keys_str[:ensemble_n_best]:
            if (
                k not in self.read_preds or self.read_preds[k][Y_ENSEMBLE] is None
            ) and self.read_losses[k]["loaded"] != 3:
                self.read_preds[k][Y_ENSEMBLE] = self._read_np_fn(k)
                # No need to load valid and test here because they are loaded
                #  only if the model ends up in the ensemble
                self.read_losses[k]["loaded"] = 1

        # return keys of self.read_losses with lowest losses
        return sorted_keys_str[:ensemble_n_best]

    def get_valid_test_preds(self, selected_keys: List[str]) -> Tuple[List[str], List[str]]:
        """
        get valid and test predictions from disc
        and store them in self.read_preds

        Parameters
        ---------
        selected_keys: list
            list of selected keys of self.read_preds

        Return
        ------
        success_keys:
            all keys in selected keys for which we could read the valid and
            test predictions
        """
        success_keys_valid = []
        success_keys_test = []

        for k in selected_keys:
            valid_fn = glob.glob(
                os.path.join(
                    glob.escape(self.backend.get_runs_directory()),
                    "%d_%d_%s"
                    % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                    "predictions_valid_%d_%d_%s.npy*"
                    % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                )
            )
            valid_fn = [vfn for vfn in valid_fn if vfn.endswith(".npy") or vfn.endswith(".npy.gz")]
            test_fn = glob.glob(
                os.path.join(
                    glob.escape(self.backend.get_runs_directory()),
                    "%d_%d_%s"
                    % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                    "predictions_test_%d_%d_%s.npy*"
                    % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                )
            )
            test_fn = [tfn for tfn in test_fn if tfn.endswith(".npy") or tfn.endswith(".npy.gz")]

            if len(valid_fn) == 0:
                pass
            else:
                if (
                    self.read_losses[k]["mtime_valid"] == os.path.getmtime(valid_fn[0])
                    and k in self.read_preds
                    and self.read_preds[k][Y_VALID] is not None
                ):
                    success_keys_valid.append(k)
                    continue
                try:
                    y_valid = self._read_np_fn(valid_fn[0])
                    self.read_preds[k][Y_VALID] = y_valid
                    success_keys_valid.append(k)
                    self.read_losses[k]["mtime_valid"] = os.path.getmtime(valid_fn[0])
                except Exception:
                    self.logger.warning(
                        "Error loading {}: {}".format(valid_fn[0], traceback.format_exc())
                    )

            if len(test_fn) == 0:
                pass
            else:
                if (
                    self.read_losses[k]["mtime_test"] == os.path.getmtime(test_fn[0])
                    and k in self.read_preds
                    and self.read_preds[k][Y_TEST] is not None
                ):
                    success_keys_test.append(k)
                    continue
                try:
                    y_test = self._read_np_fn(test_fn[0])
                    self.read_preds[k][Y_TEST] = y_test
                    success_keys_test.append(k)
                    self.read_losses[k]["mtime_test"] = os.path.getmtime(test_fn[0])
                except Exception:
                    self.logger.warning(
                        "Error loading {}: {}".format(
                            test_fn[1],
                            traceback.format_exc(),
                        )
                    )

        return success_keys_valid, success_keys_test

    def fit_ensemble(self, selected_keys: List[str]) -> Optional[EnsembleSelection]:
        """
        fit ensemble

        Parameters
        ---------
        selected_keys: list
            list of selected keys of self.read_losses

        Returns
        -------
        ensemble: EnsembleSelection
            trained Ensemble
        """

        if self.unit_test:
            raise MemoryError()

        predictions_train = [self.read_preds[k][Y_ENSEMBLE] for k in selected_keys]
        include_num_runs = [
            (
                self.read_losses[k]["seed"],
                self.read_losses[k]["num_run"],
                self.read_losses[k]["budget"],
            )
            for k in selected_keys
        ]

        # check hash if ensemble training data changed
        current_hash = "".join(
            [
                str(zlib.adler32(predictions_train[i].data.tobytes()))
                for i in range(len(predictions_train))
            ]
        )
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                "-- current performance: {}".format(
                    self.validation_performance_,
                )
            )

            return None
        self.last_hash = current_hash

        ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            loss_fn=self.loss_fn,
            loss_fn_args=self.loss_fn_args,
            random_state=self.random_state,
        )

        try:
            self.logger.debug(
                "Fitting the ensemble on {} models.".format(
                    len(predictions_train),
                )
            )
            start_time = time.time()
            ensemble.fit(predictions_train, self.y_true_ensemble, include_num_runs)
            end_time = time.time()
            self.logger.debug(
                "Fitting the ensemble took {:.2f} seconds.".format(
                    end_time - start_time,
                )
            )
            self.logger.info(str(ensemble))
            self.validation_performance_ = min(
                self.validation_performance_,
                ensemble.get_validation_performance(),
            )

        except ValueError:
            self.logger.error("Caught ValueError: {traceback.format_exc()}")
            return None
        except IndexError:
            self.logger.error("Caught IndexError: {traceback.format_exc()}")
            return None
        finally:
            # Explicitly free memory
            del predictions_train

        return ensemble

    def predict(
        self,
        set_: str,
        ensemble: AbstractEnsemble,
        selected_keys: List[str],
        n_preds: int,
        index_run: int,
    ) -> np.ndarray:
        """
        save preditions on ensemble, validation and test data on disc

        Parameters
        ----------
        set_: ["valid","test"]
            data split name
        ensemble: EnsembleSelection
            trained Ensemble
        selected_keys: list
            list of selected keys of self.read_losses
        n_preds: int
            number of prediction models used for ensemble building
            same number of predictions on valid and test are necessary
        index_run: int
            n-th time that ensemble predictions are written to disc

        Return
        ------
        y: np.ndarray
        """
        self.logger.debug(f"Predicting the {set_} set with the ensemble!")

        if set_ == "valid":
            pred_set = Y_VALID
        elif set_ == "test":
            pred_set = Y_TEST
        else:
            pred_set = Y_ENSEMBLE
        predictions = [self.read_preds[k][pred_set] for k in selected_keys]

        if n_preds == len(predictions):
            y = ensemble.predict(predictions)
            if "binary" in type_of_target(self.y_true_ensemble):
                y = y[:, 1]
            if self.SAVE2DISC:
                self.backend.save_predictions_as_txt(
                    predictions=y,
                    subset=set_,
                    idx=index_run,
                    prefix=self.dataset_name,
                    precision=8,
                )
            return y
        else:
            self.logger.info(
                f"Found inconsistent number of predictions and models ({len(predictions)} "
                f"vs {n_preds}) for subset {set_}",
            )
            return None

    def _add_ensemble_trajectory(
        self, train_pred: np.ndarray, valid_pred: np.ndarray, test_pred: np.ndarray
    ) -> None:
        """
        Records a snapshot of how the performance look at a given training
        time.

        Parameters
        ----------
        ensemble: EnsembleSelection
            The ensemble selection object to record
        valid_pred: np.ndarray
            The predictions on the validation set using ensemble
        test_pred: np.ndarray
            The predictions on the test set using ensemble

        """
        if "binary" in type_of_target(self.y_true_ensemble):
            if len(train_pred.shape) == 1 or train_pred.shape[1] == 1:
                train_pred = np.vstack(
                    ((1 - train_pred).reshape((1, -1)), train_pred.reshape((1, -1)))
                ).transpose()
            if valid_pred is not None and (len(valid_pred.shape) == 1 or valid_pred.shape[1] == 1):
                valid_pred = np.vstack(
                    ((1 - valid_pred).reshape((1, -1)), valid_pred.reshape((1, -1)))
                ).transpose()
            if test_pred is not None and (len(test_pred.shape) == 1 or test_pred.shape[1] == 1):
                test_pred = np.vstack(
                    ((1 - test_pred).reshape((1, -1)), test_pred.reshape((1, -1)))
                ).transpose()

        performance_stamp = {
            "Timestamp": pd.Timestamp.now(),
            "ensemble_optimization_loss": self.loss_fn(
                self.y_true_ensemble,
                train_pred,
                **self.loss_fn_args,
            ),
        }
        if valid_pred is not None:
            # TODO: valid_pred are a legacy from competition manager
            # and this if never happens. Re-evaluate Y_valid support
            performance_stamp["ensemble_val_loss"] = self.loss_fn(
                self.y_valid,
                valid_pred,
                **self.loss_fn_args,
            )

        # In case test_pred was provided
        if test_pred is not None:
            performance_stamp["ensemble_test_loss"] = self.loss_fn(
                self.y_test,
                test_pred,
                **self.loss_fn_args,
            )

        self.ensemble_history.append(performance_stamp)

    def _get_list_of_sorted_preds(self) -> List[Tuple[str, float, int]]:
        """
        Returns a list of sorted predictions in descending order
        Losses are taken from self.read_losses.

        Parameters
        ----------
        None

        Return
        ------
        sorted_keys: list
        """
        # Sort by loss - smaller is better!
        sorted_keys = sorted(
            [(k, v["ens_loss"], v["num_run"]) for k, v in self.read_losses.items()],
            # Sort by loss as priority 1 and then by num_run on a ascending order
            # We want small num_run first
            key=lambda x: (x[1], x[2]),
        )
        return sorted_keys

    def _delete_excess_models(self, selected_keys: List[str]) -> None:
        """
        Deletes models excess models on disc. self.max_models_on_disc
        defines the upper limit on how many models to keep.
        Any additional model with a worst loss than the top
        self.max_models_on_disc is deleted.

        """

        # Loop through the files currently in the directory
        for pred_path in self.y_ens_files:

            # Do not delete candidates
            if pred_path in selected_keys:
                continue

            if pred_path in self._has_been_candidate:
                continue

            match = self.model_fn_re.search(pred_path)
            if match is None:
                raise ValueError(f"Could not interpret pred_path={pred_path}")
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            # Do not delete the dummy prediction
            if _num_run == 1:
                continue

            numrun_dir = self.backend.get_numrun_directory(_seed, _num_run, _budget)
            try:
                os.rename(numrun_dir, numrun_dir + ".old")
                shutil.rmtree(numrun_dir + ".old")
                self.logger.info("Deleted files of non-candidate model {}".format(pred_path))
                self.read_losses[pred_path]["disc_space_cost_mb"] = None
                self.read_losses[pred_path]["loaded"] = 3
                self.read_losses[pred_path]["ens_loss"] = np.inf
            except Exception as e:
                self.logger.error(
                    "Failed to delete files of non-candidate model {} due"
                    " to error {}".format(
                        pred_path,
                        e,
                    )
                )

    def _read_np_fn(self, path: str) -> np.ndarray:

        # Support for string precision
        precision = int(self.precision)

        if path.endswith("gz"):
            fp = gzip.open(path, "rb")
        elif path.endswith("npy"):
            fp = open(path, "rb")
        else:
            raise ValueError("Unknown filetype {path}")
        if precision == 16:
            predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float16)
        elif precision == 32:
            predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float32)
        elif precision == 64:
            predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float64)
        else:
            predictions = np.load(fp, allow_pickle=True)
        fp.close()
        return predictions
