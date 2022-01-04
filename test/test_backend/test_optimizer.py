from pathlib import Path

import pytest

from automl_common.backend import Context, Optimizer


def test_construction(tmpdir: Path, context: Context):
    """
    Parameters
    ----------
    tmpdir: Path
        Path to an existing tmpdir

    context: Context
        A context tp access the filesystem with
    """
    Optimizer(dir=tmpdir, context=context)
