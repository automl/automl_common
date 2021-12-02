from typing import Any
import pytest

import numpy as np

from pathlib import Path

from automl_common.backend import Context, Run


def test_construction(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to an existing tmpdir

    context: Context
        A context to access the filesystem with

    Expects
    -------
    * The properties of the run should be set
    * It should not `exist` yet
    """
    id = "hello"
    run_folder = context.join(tmpdir, id)
    run = Run(id=id, dir=run_folder, context=context)

    assert run.id == id
    assert run.dir == run_folder
    assert run.context == context
    assert not run.exists


def test_construction_with_existing_dir(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to an existing tmpdir

    context: Context
        A context to access the filesystem with

    Expects
    -------
    * The properties of the run should be set
    * It should `exist`
    """
    id = "hello"
    run_folder = context.join(tmpdir, id)
    context.mkdir(run_folder)

    run = Run(id=id, dir=run_folder, context=context)

    assert run.id == id
    assert run.dir == run_folder
    assert run.context == context
    assert run.exists


def test_model_path_property(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * The run should have it's model path in an expected place
    """
    expected = run.context.join(run.dir, "model")
    assert run.model_path == expected


@pytest.mark.parametrize("model", [1, {"hello": "world"}, (1, 2, 3)])
def test_save_model(run: Run, model: Any):
    """
    Parameters
    ----------
    run: Run
        The run to check

    model: Any
        The 'model' to save

    Expects
    -------
    * Should be able to save the model to disk
    """
    run.save_model(model)
    assert run.context.exists(run.model_path)


@pytest.mark.parametrize("model", [1, {"hello": "world"}, (1, 2, 3)])
def test_has_model_when_it_has_one_saved(run: Run, model: Any):
    """
    Parameters
    ----------
    run: Run
        The run to check

    model: Any
        The 'model' to save

    Expects
    -------
    * Should recognize it has a model saved
    """
    run.save_model(model)
    assert run.has_model()


def test_has_model_when_it_has_none_saved(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should recognize it has no model saved
    """
    assert not run.has_model()


@pytest.mark.parametrize("model", [1, {"hello": "world"}, (1, 2, 3)])
def test_load_model_when_has_model(run: Run, model: Any):
    """
    Parameters
    ----------
    run: Run
        The run to check

    model: Any
        The 'model' to save

    Expects
    -------
    * Should be able to load a model saved to disk
    * The loaded model should be the same as the one saved
    """
    run.save_model(model)
    assert run.has_model()

    loaded = run.model()
    assert loaded == model


def test_load_model_when_not_has_model(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should raise a RuntimeError if the model doesn't exist yet
    """
    assert not run.has_model()
    with pytest.raises(RuntimeError):
        run.model()


@pytest.mark.parametrize("prefix", ["train", "test", "_other", "x.y"])
def test_predictions_path_has_prefix(run: Run, prefix: str):
    """
    Parameters
    ----------
    run: Run
        The run to check

    prefix: str
        The prefix to add to the str

    Expects
    -------
    * Path should be in the run's dir and include the prefix with .npy
      as we save with numpy.
    """
    path = run.predictions_path(prefix)
    expected = run.context.join(run.dir, f"{prefix}_predictions.npy")
    assert path == expected


def test_has_predictions_when_no_predictions(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should show that it has no predictions
    """
    assert not run.has_predictions("any")


def test_has_predictions_when_has_predictions_but_wrong_prefix(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should return that it does not have predictions with prefix "two".
    """
    x = [1, 2, 3]
    run.save_predictions(x, "one")
    assert not run.has_predictions("two")


def test_has_predictions_when_has_predictions_with_correct_prefix(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should return that it has predictions with prefix "one"
    """
    x = [1, 2, 3]
    run.save_predictions(x, "one")
    assert run.has_predictions("one")


def test_save_predictions(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should show that it saved predictions to the expected file
    """
    x = [1, 2, 3]
    run.save_predictions(x, "one")
    expected_path = run.predictions_path("one")
    assert run.context.exists(expected_path)


def test_save_predictions_with_different_prefixes(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should save both predictions to file and give them seperate filenames with
        their prefix
    """
    x = [1, 2, 3]

    run.save_predictions(x, "one")

    expected_path_1 = run.predictions_path("one")
    assert run.context.exists(expected_path_1)

    run.save_predictions(x, "two")

    expected_path_2 = run.predictions_path("two")
    assert run.context.exists(expected_path_2)

    assert expected_path_1 != expected_path_2


def test_load_predictions_when_none_exist(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should raise an Error as it has nothing to load
    """
    with pytest.raises(FileNotFoundError):
        run.predictions("one")


def test_load_predictions_when_exists(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should load the predictions and be same as those that were stored
    """
    x = np.asarray([1, 2, 3])
    run.save_predictions(x, "one")

    loaded = run.predictions("one")
    np.testing.assert_equal(x, loaded)


def test_load_predictions_when_both_exist(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should load predictions with seperate prefixes and be the same as those stored
    """
    one = np.asarray([1, 1, 1])
    run.save_predictions(one, "one")

    two = np.asarray([2,2,2])
    run.save_predictions(two, "two")

    loaded_one = run.predictions("one")
    loaded_two = run.predictions("two")

    # Numpy has no negation of it's testing assertions
    # https://stackoverflow.com/a/62504156
    with pytest.raises(AssertionError):
        np.testing.assert_equal(loaded_one, loaded_two)

    np.testing.assert_equal(one, loaded_one)
    np.testing.assert_equal(two, loaded_two)


def test_setup_when_not_exists(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should create the dir and mark itself as existing
    """
    run.setup()
    assert run.context.exists(run.dir)
    assert run.exists


def test_setup_when_exists(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should raise an error that setup was called when it already exists
    """
    run.setup()

    with pytest.raises(RuntimeError):
        run.setup()


def test_equality_with_self(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should be equal to itself
    * Should not be != to itself
    """
    assert run == run
    assert run.__eq__(run)
    assert not run != run


def test_equality_with_other_equivalent_run(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should equal the other run when id and dir is the same
    """
    other = Run(id=run.id, dir=run.dir, context=run.context)
    assert run == other
    assert run.__eq__(other)


def test_equality_with_other_different_run(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should not equal a run with a different id
    """
    other = Run(id="nope", dir=run.dir, context=run.context)
    assert not run == other
    assert not run.__eq__(other)
    assert run != other


def test_equality_works_with_str_conversion(run: Run):
    """
    Parameters
    ----------
    run: Run
        The run to check

    Expects
    -------
    * Should equal the other run if the only difference is a str conversion of the id
    """
    other = Run(id=str(run.id), dir=run.dir, context=run.context)
    assert run == other
    assert run.__eq__(other)


@pytest.mark.parametrize("obj", [ [1,2,3], {"hello": "world"}, "hi" ])
def test_equality_with_different_type(run: Run, obj: Any):
    """
    Parameters
    ----------
    run: Run
        The run to check

    obj: Any
        The thing to compare to

    Expects
    -------
    * Equality check between a run and a different type should raise a NotImplementedError
    """
    with pytest.raises(NotImplementedError):
        run == obj

    with pytest.raises(NotImplementedError):
        run.__eq__(obj)

    with pytest.raises(NotImplementedError):
        run != obj
