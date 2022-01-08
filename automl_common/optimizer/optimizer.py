from abc import ABC, abstractmethod

from automl_common.backend import Backend


class Optimizer(ABC):
    def __init__(self, backend: Backend):
        self.backend = backend
