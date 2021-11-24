from typing import TypeVar, Generic, cast

import pickle

DM = TypeVar("T")

class DataManager(Generic[DM]):

    def __init__(self, dir: str, context: Context):
        self.dir = dir
        self.context = context

    @property
    def datamanager_path(self) -> str:
        return self.context.join(self.dir, "datamanager.pkl")

    def save(self, datamanager: DM) -> None:
        with open(self.datamanager_path, "wb") as f:
            pickle.dump(datamanger, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> DM:
        with open(self.datamanager_path, "rb") as f:
            datamanger = pickle.load(datamanger, f, protocol=pickle.HIGHEST_PROTOCOL)

        return cast(DM, datamanger)
