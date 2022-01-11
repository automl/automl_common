from pathlib import Path

from automl_common.backend.contexts.context import Context


class AWSPath(Path):
    """A special Path object for AWS if needed"""

    def __init__(self) -> None:
        raise NotImplementedError()


class AWSContext(Context):
    """A Context for AWS ... example of what other contexts could exist"""

    def __init__(self, some_key: str):
        raise NotImplementedError()

    @classmethod
    def as_path(cls, path: str) -> AWSPath:
        """Would use the special path for AWS"""
        raise NotImplementedError()
