from typing import List, TypeVar

from sklearn.utils.validation import check_is_fitted

from automl_common.sklearn.ensemble.base import Ensemble
from automl_common.sklearn.model import Predictor

PT = TypeVar("PT", bound=Predictor)


class SingleEnsemble(Ensemble[PT]):
    """An ensemble that select the single best.

    Parameters
    ----------
    metric : (np.ndarray, np.ndarray) -> Orderable
        A metric to evaluate models with. Should return an Orderable result

    select: "min" | "max"
        How to order results of the metric

    random_state : Optional[Union[int, np.random.RandomState]] = None
        The random_state to use for breaking ties

    model_store : Optional[ModelStore[PT]] = None
        A store of models to use during fit
    """

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
    def id(self) -> str:
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
        return super()._fit_attributes() + ["random_state_", "model_id_"]
