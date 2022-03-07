from typing_extensions import Literal

from pathlib import Path


def sizeof(path: Path, unit: Literal["B", "KB", "MB", "GB", "TB"] = "B") -> int:
    """Get the size of the path, whether it be a file or directory

    Parameters
    ----------
    path : Path
        The path to get the size of

    unit : Literal["B", "KB", "MB", "GB", "TB"] = "B"
        What unit to get the answer in.

    Returns
    -------
    int
        The size of the file/dir at path
    """
    # https://stackoverflow.com/a/1392549/5332072
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

    conversion = {
        "B": 1,
        "KB": 10,
        "MB": 20,
        "GB": 30,
        "TB": 40,
    }
    return round(size / (2 ** conversion["unit"]))
