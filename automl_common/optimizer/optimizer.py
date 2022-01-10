from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pathlib import Path

from automl_common.backend import Backend
from automl_common.backend.contexts import OSContext

Handler = Callable[..., Any]


class Optimizer(ABC):
    """An optimizer over a ConfigurationSpace

    An optimizer can provide a list of events that users can subscribe
    handlers to.

    An optimizer can provide capabilities from a set list:
    * 'external-backend' - Will not write to the local filesystem
    * 'intertupable' - Has a mechanism to interup it
    """

    def __init__(self, backend: Backend):
        """
        Parameters
        ----------
        backend: Backend
            The backend the optimizer has access to
        """
        # If we have a non local backend and the optimizer does not support it
        if not (self.supports("external-backend") and isinstance(backend, OSContext)):
            raise ValueError(
                f"Optimizer {self.__class__.__name__} does not support an external "
                "backend. Please use the `OSContext` for the backend."
            )

        self.backend = backend

        # Delayed construction, in case subclass needs to dynamically populate @events
        self._handlers: Optional[Dict[str, List[Handler]]] = None

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

    @abstractmethod
    @property
    def output_dir(self) -> Path:
        """Where the output of the optimizer will be stored"""
        ...

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

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in self.handlers[event]:
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
        if event not in self.handlers:
            self.handlers[event] = [handler]
        else:
            self.handlers[event].append(handler)
