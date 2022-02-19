from typing import Callable

from pathlib import Path

import numpy as np

from automl_common.backend.stores.predictions_store import PredictionsStore

from pytest_cases import parametrize


@parametrize("bad_filename", ["predictions_not.txt", "hello", "here.npy"])
def test_iter_with_other_non_predicitons_in_same_dir(
    path: Path,
    make_predictions_store: Callable[..., PredictionsStore],
    bad_filename: str,
) -> None:
    """
    Parameters
    ----------
    path: Path
        Path to the store

    make_predictions_store: Callable
        A factory to create a predictions store

    bad_filename: str
        A file that predictions store should not pick up on

    Expects
    -------
    * Should only pick up files starting with `predictions_{key}.npy`
    """
    preds = {key: np.array([1]) for key in ["train", "test", "val"]}

    store = make_predictions_store(path, preds)

    bad_path = store.dir / bad_filename
    bad_path.write_text("hello")

    keys = list(store.keys())

    # Shouldn't see {bad_filename} in here
    assert bad_filename not in keys
    assert set(keys) == set(preds.keys())
