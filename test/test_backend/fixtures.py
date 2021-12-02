import pytest

from pytest_lazyfixture import lazy_fixture

from pathlib import Path

from automl_common.backend import Context, LocalContext, Run, Runs, DataManager, Ensemble


@pytest.fixture(scope="function")
def local_context() -> LocalContext:
    return LocalContext()


@pytest.fixture(scope="function")
def other_context() -> LocalContext:
    return LocalContext()


@pytest.fixture(scope="function", params=[lazy_fixture("local_context")])
def context(request) -> Context:
    """All Contexts collected together"""
    return request.param


@pytest.fixture(scope="function")
def datamanager(tmpdir: Path, context: Context) -> DataManager:
    return DataManager(dir=tmpdir, context=context)


@pytest.fixture(scope="function")
def run_int(tmpdir: Path, context: Context) -> Run:
    id = 1
    path = context.join(tmpdir, str(id))
    return Run(id=id, dir=path, context=context)


@pytest.fixture(scope="function")
def run_tuple(tmpdir: Path, context: Context) -> Run:
    id = (1, 1)
    path = context.join(tmpdir, str(id))
    return Run(id=id, dir=path, context=context)


@pytest.fixture(scope="function", params=[lazy_fixture("run_int"), lazy_fixture("run_tuple")])
def run(request) -> Run:
    return request.param


@pytest.fixture(scope="function")
def ensemble_int(tmpdir: Path, context: Context) -> Ensemble:
    id = 1
    path = context.join(tmpdir, str(id))
    return Ensemble(id=id, dir=path, context=context)


@pytest.fixture(scope="function")
def ensemble_tuple(tmpdir: Path, context: Context) -> Ensemble:
    id = (1, 1)
    path = context.join(tmpdir, str(id))
    return Ensemble(id=id, dir=path, context=context)


@pytest.fixture(
    scope="function",
    params=[lazy_fixture("ensemble_int"), lazy_fixture("ensemble_tuple")]
)
def ensemble(request) -> Ensemble:
    return request.param


@pytest.fixture(scope="function")
def runs(request, tmpdir: Path, context: Context) -> Runs:
    """Creates Runs objects, populating with id's passed as parameters

    https://docs.pytest.org/en/latest/example/parametrize.html#indirect-parametrization
    ```
    # How to use

    @pytest.mark.parametrize("runs", [[1,2,3], [], [(1,2), (3,4)]], indirect=True)
    def test_func(runs: Runs):
        ...
    ```

    Parameters
    ----------
    request.param: List[Any]
        The id's of runs to populate

    Returns
    -------
    Runs
        Returns the Runs object along with any runs it should contain
    """
    ids = request.param
    print(ids)
    runs = [Run(id, context.join(tmpdir, str(id)), context) for id in ids]

    for run in runs:
        run.save_model("this string is a model")
        run.save_predictions([1, 1, 1], "train")

    return Runs(dir=tmpdir, context=context)
