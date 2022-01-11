from pytest_cases import parametrize

from automl_common.backend.stores import PredictionsStore


@parametrize("bad_filename", ["predictions_not.txt", "hello", "here.npy"])
def test_iter(predictions_store: PredictionsStore, bad_filename: str) -> None:
    """
    Parameters
    ----------
    predictions_store: PredictionsStore
        The predictions store

    bad_filename: str
        A file that predictions store should not pick up on

    Expects
    -------
    * Should only pick up files starting with `predictions_{key}.npy`
    """
    context = predictions_store.context
    dummy = predictions_store.dir / bad_filename
    with context.open(dummy, "w") as f:
        f.write("hello")

    keys = list(iter(predictions_store))

    assert set(keys) == set(["train", "test", "val"])
