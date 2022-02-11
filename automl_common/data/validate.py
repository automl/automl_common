import numpy as np


def jagged(x: np.ndarray) -> bool:
    """Tests if a numpy array is jagged.

    Note
    ----
    This relies on the fact that each row will be converted to
    a list with ``x`` being made 1-dimensional and dtype object.

    If all these conditions are true, we are forced to iterate until
    we find a jagged row.

    Parameters
    ----------
    x : np.ndarray
        The array to check

    Returns
    -------
    bool
        Whether the array is jagged or not
    """
    return (
        x.dtype == "object"
        and x.ndim == 1
        and isinstance(x[0], list)
        and not all(len(row) == x[0] for row in x)
    )  # pragma: no cover, not sure why it's no covered
