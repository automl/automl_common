from automl_common.backend.context import Context

from Pathlib import Path


class AWSPath(Path):
    """A special Path object for AWS if needed"""

    pass


class AWSContext(Context):
    """A Context for AWS ... example of what other contexts could exist"""

    def __init__(self, some_key: str):
        raise NotImplementedError()

    def as_path(self, path: str) -> AWSPath:
        """Would use the special path for AWS"""
        raise NotImplementedError()
