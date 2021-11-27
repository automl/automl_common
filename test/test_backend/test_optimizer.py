import pytest

from pathlib import Path

from automl_common.backend.optimizer import Optimizer
from automl_common.backend.context import Context

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
