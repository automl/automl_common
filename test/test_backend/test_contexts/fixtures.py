from pytest_cases import fixture, fixture_ref, parametrize

from automl_common.backend.contexts import Context, OSContext


@fixture(scope="function")
def os_context() -> OSContext:
    """
    Returns
    -------
    OSContext
        An OSContext
    """
    return OSContext()


@fixture(scope="function")
@parametrize("context_impl", [fixture_ref(os_context)])
def context(context_impl: Context) -> Context:
    """Used to collect different context implementations and test the all

    Parameters
    ----------
    context_impl: Context
        The context to forward on

    Returns
    -------
    Context
        The forwarded context
    """
    return context_impl
