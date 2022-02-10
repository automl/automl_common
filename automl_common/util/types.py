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
