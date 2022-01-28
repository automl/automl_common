from typing import Any
from typing_extensions import Protocol  # TODO: Update with Python 3.8


class SupportsEqualty(Protocol):
    def __eq__(self, o: Any) -> bool:
        ...
