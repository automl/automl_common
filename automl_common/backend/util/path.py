from pathlib import Path


def rmtree(path: Path) -> None:
    """Provides an os independant rmdir based purely on the path

    Parameters
    ----------
    path: Path
        The path to delete
    """
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rmtree(child)
    path.rmdir()
