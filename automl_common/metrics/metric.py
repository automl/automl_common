from typing import Protocol

import numpy as np


class MetricProtocol(Protocol):
    """Describes an interface for a metric"""

    def __call__(self, predicitons: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        """
        Parameters
        ----------
        predicitons: np.ndarray
            The predictions of a model

        targets: np.ndarray
            The targets to compare to

        **kwargs
            Any arguments tht will be forwarded to implementers
        """
        ...

    def score(self, predicitons: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        """
        Parameters
        ----------
        predicitons: np.ndarray
            The predictions of a model

        targets: np.ndarray
            The targets to compare to

        **kwargs
            Any arguments tht will be forwarded to implementers
        """
        ...

    def loss(self, predicitons: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        """
        Parameters
        ----------
        predicitons: np.ndarray
            The predictions of a model

        targets: np.ndarray
            The targets to compare to

        **kwargs
            Any arguments tht will be forwarded to implementers
        """
        ...
