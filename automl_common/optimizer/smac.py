from automl_common.backend import Backend
from automl_common.optimizer import Optimizer


class SMACOptimizer(Optimizer):
    def __init__(self, backend: Backend):
        super().__init__(backend)
