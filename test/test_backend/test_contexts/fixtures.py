from pytest_cases import fixture

from automl_common.backend.contexts import OSContext


@fixture(scope="function")
def os_context() -> OSContext:
    """
    Returns
    -------
    OSContext
        An OSContext

    """
    return OSContext()
