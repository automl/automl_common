from typing import Any, Protocol


class SupportsEqualty(Protocol):
    def __eq__(self, o: Any) -> bool:
        ...
