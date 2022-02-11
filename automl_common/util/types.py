from typing import Any
from typing_extensions import Protocol  # TODO, change in python 3.8


# For support min and max which require SupportsRichComparison, simplied to Orderable
# https://github.com/python/typeshed/blob/master/stdlib/_typeshed/__init__.pyi#L54
class Orderable(Protocol):
    def __eq__(self, other: Any) -> bool:
        ...

    def __lt__(self, other: Any) -> bool:
        ...

    def __gt__(self, other: Any) -> bool:
        ...


class EqualityMixin:  # pragma: no cover
    """Add basic equality checkng to a class

    https://stackoverflow.com/a/390511/5332072
    https://stackoverflow.com/a/390640/5332072
    """

    def __eq__(self, other: Any) -> bool:
        if not type(self) is type(other):
            return NotImplemented
        else:
            return self.__dict__ == other.__dict__
