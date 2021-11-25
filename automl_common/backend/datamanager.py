from typing import TypeVar, Generic, cast

import pickle

DM = TypeVar("T")

class DataManager(Generic[DM]):
    """An interface to the datamanager part of the backend"""

    def __init__(self, dir: str, context: Context):
        """
        Parameters
        ----------
        dir: str
            The directory where data is stored

        context: Context
            The context to perform file operations on
        """
        self.dir = dir
        self.context = context

    @property
    def datamanager_path(self) -> str:
        """The path to the datamanager"""
        return self.context.join(self.dir, "datamanager.pkl")

    def save(self, datamanager: DM) -> None:
        """Save a Datamanager

        Parameters
        ----------
        datamanger: DM
            Save a datamanager
        """
        with open(self.datamanager_path, "wb") as f:
            pickle.dump(datamanger, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> DM:
        """Load a datamanger

        Returns
        -------
        DM
            Load a datamanager into memory
        """
        with open(self.datamanager_path, "rb") as f:
            datamanger = pickle.load(datamanger, f, protocol=pickle.HIGHEST_PROTOCOL)

        return cast(DM, datamanger)
