from abc import ABC, abstractmethod
from typing import Callable, Dict, List

from automl_common.backend import Backend

Handler = Callable[..., ...]


class Optimizer(ABC):
    """An optimizer over a ConfigurationSpace

    An optimizer can provide a list of events that users can subscribe
    handlers to.

    An optimizer can provide capabilities from a set list, if doing
    so it must override the corresponding method:
    * "``warmstart``" - The optimizer can be warm started
    """

    def __init__(self, backend: Backend):
        """
        Parameters
        ----------
        backend: Backend
            The backend the optimizer has access to
        """
        self.backend = backend

        # Delayed construction, in case subclass needs to dynamically populate @events
        self._handlers = None

    @abstractmethod
    def optimize(self, *args, **kwargs) -> List[str]:
        """Optimizes over a configuration space.

        Returns
        -------
        List[str]
            A list of model ids
        """
        ...

    @property
    def handlers(self) -> Dict[str, List[Handler]]:
        """The handlers listed for this optimized

        Returns
        -------
        Dict[str, List[Handler]]
            Lists of handlers indexed by the event they're subscribed to
        """
        if self._handlers is None:
            self._handlers = {event: [] for event in self.events}

        return self._handlers

    @abstractmethod
    @property
    def events(self) -> List[str]:
        """A list of events the optimizer will emit"""
        ...

    @abstractmethod
    @property
    def capabilities(self) -> List[str]:
        """The List of capabilities provided by the Optimizer

        Returns
        -------
        List[str]
            A list of capabilities provided by the optimizer
        """
        ...

    def warmstart(self, *args, **kwargs) -> None:
        """Warmstart the optimizer"""
        raise NotImplementedError("Optimizer does not support 'warmstart'")

    def supports(self, capability: str) -> bool:
        """Check if the Optimizer supports a given capability

        Parameters
        ----------
        capability: str
            The capability to check

        Returns
        -------
        bool
            Whether this optimizer supports the given capability
        """
        return capability in self.capabilities

    def _emit(self, event: str, *args, **kwargs) -> None:
        for handler in self._handlers.get(event, []):
            handler(*args, **kwargs)

    def on(self, event: str, handler: Handler) -> None:
        """Register a handler that handlers an event emitted with 'event'

        Parameters
        ----------
        event: str
            The event to subscribe a handler to

        handler: Handler
            A handler for this event
        """
        if event not in self._handlers:
            self._handlers[event] = [handler]
        else:
            self._handlers[event] += handler
