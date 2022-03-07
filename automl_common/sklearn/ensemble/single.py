from typing import List, TypeVar

from sklearn.utils.validation import check_is_fitted

from automl_common.sklearn.ensemble.base import Ensemble
from automl_common.sklearn.model import Predictor

PT = TypeVar("PT", bound=Predictor)
ID = TypeVar("ID")


class SingleEnsemble(Ensemble[ID, PT]):
    """An ensemble that select the single best."""

    @property
    def model(self) -> PT:
        """Get the model of this ensemble

        Returns
        -------
        PT
            Returns the model of this ensemble.

        Raises
        ------
        NotFittedError
            If the ensemble has not been fit yet
        """
        return self.__getitem__(self.id)

    @property
    def id(self) -> ID:
        """Get the id of the selected model
        Returns
        -------
        str
            The id of the selected model

        Raises
        ------
        NotFittedError
            If the ensemble has not been fit yet
        """
        check_is_fitted(self)
        return self.model_id_  # type: ignore

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        return super()._fit_attributes() + ["model_id_"]
