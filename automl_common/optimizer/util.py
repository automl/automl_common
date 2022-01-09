from typing import Callable, Optional

from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

RunCompleteHandler = Callable[[SMBO, RunInfo, RunValue, float], Optional[bool]]


class RunCompleteWrap(IncorporateRunResultCallback):

    def __init__(self, handler: RunCompleteHandler):
        self._handler = handler

    def __call__(self, *args, **kwargs) -> bool:
        """Forwards args to registered handler"""
        return self._handler(*args, **kwargs)
